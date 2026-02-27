import logging
import warnings
from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from analysislib.common.image import ROI, Image

logger = logging.getLogger(__name__)


class TweezerFinder:
    def __init__(self, image: Union[Image, ArrayLike]):
        if isinstance(image, Image):
            self.image = image
        else:
            self.image = Image(np.array(image))

    def detect_rois_by_roi_number(
        self,
        roi_number: int,
        neighborhood_size: int = 5,
        detection_threshold: float = 25,
        roi_size: int = 5,
        search_step: float = 0.5,
        restricted_ROI = None,
    ):
        site_rois = self.detect_rois(
            neighborhood_size=neighborhood_size,
            detection_threshold=detection_threshold,
            roi_size=roi_size,
            restricted_ROI=restricted_ROI,
        )

        site_count_difference_init = len(site_rois) - roi_number
        if site_count_difference_init == 0:
            return site_rois

        current_threshold = detection_threshold
        init_difference_sign = +1 if site_count_difference_init > 0 else -1
        threshold_step = search_step * init_difference_sign
        while (len(site_rois) - roi_number) * init_difference_sign > 0:
            current_threshold += threshold_step
            print(f"Attempting threshold {current_threshold}")
            site_rois = self.detect_rois(
                neighborhood_size,
                current_threshold,
                roi_size,
                restricted_ROI=restricted_ROI,
            )

        site_rois = self.remove_overlapping_rois(site_rois, min_distance=roi_size)
        print(f"Exactly {len(site_rois)} sites found")

        if len(site_rois) > roi_number:
            last_index = roi_number - len(site_rois)
            site_rois = site_rois[:last_index]

        print(site_rois)
        return site_rois

    def detect_rois_by_contours(
            self,
            roi_number: int,
            roi_size: int,
            restriction_roi: ROI = ROI(xmin=0, xmax=2048, ymin=0, ymax=2048),
            blur_block: int = 3,
            blur_width: float = 1,
            block_size: int = 11,
            relative_threshold: float = 4,
            affine_transform = lambda x: x,
    ) -> list[ROI]:
        """
        Detect bright spots using a contour-based approach with OpenCV.
        The image is blurred to reduce noise, then thresholded.
        For robustness against local variations in background brightness,
        this thresholding is adaptive (i.e. determined based on average local brightness).
        Resulting contours are then filtered and ROIs centered on each are produced as output.
        
        Parameters
        ----------
        roi_number: int
            Expected number of bright spots.
        roi_size: int
            Size of resulting ROIs.
        restriction_roi: ROI
            Region to restrict ROI centers to.
        blur_block: int
            Size of block for Gaussian blurring during preprocessing.
        blur_width: float
            Characteristic radius for Gaussian blurring.
        block_size: int
            Size of block for determining adaptive threshold,
            which is a Gaussian weighted average within a (block_size, block_size)
            region around each pixel, with 
        relative_threshold: float
            Value *above* weighted mean for adaptive thresholding of pixel values,
            in scaled units (i.e. after applying affine_transform).
        affine_transform: optional
            Affine function for converting blurred image pixel values
            to a value in [0, 255] to better span 8-bit space
            (adaptive thresholding requires 8-bit image values).
        """
        import cv2  # type:ignore

        from analysislib.common.contour import Contour

        if blur_block < 0 or (blur_block != 0 and blur_block % 2 != 1):
            raise ValueError(f'{blur_block=} should be zero or a positive odd integer')
        if blur_width <= 0:
            raise ValueError(f'{blur_width=} should be positive')
        if block_size < 0 or block_size % 2 != 1:
            raise ValueError(f'{block_size=} should be a positive odd integer')

        image_array_blurred = cv2.GaussianBlur(
            self.image.subtracted_array,
            (blur_block, blur_block),  # kernel size
            blur_width,  # Gaussian width
        )

        min_blurred_value, max_blurred_value = image_array_blurred.min(), image_array_blurred.max()
        if affine_transform(min_blurred_value) >= affine_transform(max_blurred_value):
            raise ValueError('affine_transform must be order-preserving!')
        if affine_transform(max_blurred_value) > 255 or affine_transform(min_blurred_value) < 0:
            warnings.warn(
                'affine_transform does not keep all pixel values between 0 and 255; '
                f'(min, max) = ({min_blurred_value:.3f}, {max_blurred_value:.3f})',
                UserWarning,
            )

        # we need 8-bit images for the following adaptive threshold step
        image_array_u8 = (affine_transform(image_array_blurred)).astype('uint8')
        thresholded = cv2.adaptiveThreshold(
            image_array_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -relative_threshold,
        )
        contours_raw, hierarchy = cv2.findContours(
            thresholded,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = (Contour(contour_raw) for contour_raw in contours_raw)
        contours_filtered = sorted(
            (
                contour
                for contour in contours
                if (
                    contour.area > 0 and
                    restriction_roi.contains(
                        contour.centroid[0],
                        contour.centroid[1] + self.image.yshift,
                    )
                )
            ),
            key=(lambda contour: contour.area),
            reverse=True,
        )
        logger.info(f'Identified {len(contours_filtered)} potential sites')
        if len(contours_filtered) < roi_number:
            warnings.warn(f'Did not find desired number ({roi_number}) of sites.')
        elif len(contours_filtered) > roi_number:
            logger.info(f'Keeping largest {roi_number} sites by area.')
        site_contours = contours_filtered[:roi_number]

        site_rois = [
            ROI.from_center(
                center_x=contour.centroid[0],
                center_y=(self.image.yshift + contour.centroid[1]),
                size=roi_size,
            )
            for contour in site_contours
        ]
        # sort ROIs by decreasing x coordinate
        site_rois.sort(key=(lambda roi: roi.xmin), reverse=True)

        self.image_blurred = image_array_blurred
        self.image_thresholded = thresholded
        self.site_contours = site_contours
        self.rois = site_rois

        return site_rois

    def remove_overlapping_rois(self, rois, min_distance):
        """
        Remove ROIs whose centers are closer than min_distance.
        Args:
            rois: list of ROI objects or tuples (x, y, ...)
            min_distance: minimum allowed distance between ROI centers
        Returns:
            Filtered list of ROIs
        """
        filtered = []
        centers = []
        for roi in rois:
            # Extract center coordinates depending on your ROI format
            if hasattr(roi, 'x') and hasattr(roi, 'y'):
                cx, cy = roi.x, roi.y
            else:
                cx, cy = roi[0], roi[1]  # adjust if your ROI is a tuple/list
            if all(np.hypot(cx - x0, cy - y0) >= min_distance for x0, y0 in centers):
                filtered.append(roi)
                centers.append((cx, cy))
        return filtered

    def detect_rois(self, neighborhood_size: int, detection_threshold: float, roi_size: int, restricted_ROI = None,):
        return self.image.detect_site_rois(neighborhood_size, detection_threshold, roi_size, restricted_ROI = restricted_ROI)

    def weight_functions(self, rois: Sequence[ROI], background_subtract: bool = False):
        img = self.image if background_subtract else self.image.raw_image()
        return [
            img.roi_view(roi) / img.roi_sum(roi) * roi.pixel_area
            for roi in rois
        ]

    def plot_contour_site_detection(
            self,
            fig: Figure,
            ylims: Union[tuple[int, int], Literal['full', 'sites']] = 'sites',
            label_sites: int = 5,
            label_displacement: tuple[float, float] = (0, -8),
            **kwargs,
    ):
        axs = fig.subplots(nrows=3)
        # print(self.rois)

        if len(self.rois) == 0:
            rois_bbox = ROI(500, 1500, 500, 1500)
        else:
            rois_bbox = ROI.bounding_box(self.rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)

        if ylims == 'full':
            plot_ylims = (
                self.image.yshift,
                self.image.yshift + self.image.height,
            )
        elif ylims == 'sites':
            plot_ylims = (rois_bbox.ymin - padding, rois_bbox.ymax + padding)
        else:
            plot_ylims = ylims

        view_roi = ROI.from_roi_xy(atom_roi_xlims, plot_ylims)

        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=np.max(self.image.subtracted_array),
        )
        imshow_kw = raw_img_color_kw | kwargs

        im = self.image.imshow_view(
            view_roi,
            ax=axs[0],
            **imshow_kw,
        )
        fig.colorbar(im, ax=axs)

        axs[1].set_title('Blurred image')
        # print(self.image_blurred.shape)

        plot_extent = np.array([view_roi.xmin, view_roi.xmax, view_roi.ymax, view_roi.ymin]) - 0.5

        image_slice = (
            slice(view_roi.ymin - self.image.yshift, view_roi.ymax - self.image.yshift),
            slice(view_roi.xmin, view_roi.xmax),
        )

        axs[1].imshow(
            self.image_blurred[image_slice],
            extent=plot_extent,
            **imshow_kw,
        )

        axs[2].set_title('Adaptive threshold')
        axs[2].imshow(
            self.image_thresholded[image_slice],
            extent=plot_extent,
            cmap='Greys_r',
        )
        
        for ax in [axs[0], axs[2]]:
            ROI.plot_rois(
                ax,
                self.rois,
                label_sites=label_sites,
                label_displacement=label_displacement,
            )

        centroids = np.array([contour.centroid for contour in self.site_contours])

        # print(centroids + [0, self.image.yshift])
        # print(f'{self.image.yshift=}')
        
        centroid_x, centroid_y = centroids.T
        axs[2].scatter(
            centroid_x,
            centroid_y + self.image.yshift,
            s=1,
            color='red',
        )
