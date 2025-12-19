import warnings
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure

from analysislib.common.analysis_config import kinetix_system
from analysislib.common.image import ROI, Image
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.tweezer_preproc import TweezerPreprocessor


class TweezerFinder:
    def __init__(self, images: Sequence[Image]):
        self.images = images
        self.averaged_image = Image.mean(images)
        self.avg_1st_img = Image.mean(images[::2])
        self.avg_2nd_img = Image.mean(images[1::2])
        # Need to include more images if we take more images per shot

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

        if len(site_rois)>roi_number:
            last_index = roi_number - len(site_rois)
            site_rois = site_rois[:last_index]

        print(site_rois)
        return site_rois

    def detect_rois_by_contours(self, roi_number: int, roi_size: int) -> list[ROI]:
        import cv2

        from analysislib.common.contour import Contour

        image_array_blurred = cv2.GaussianBlur(
            self.averaged_image.subtracted_array,
            (3, 3),  # kernel size
            1,  # Gaussian width
        )
        image_array_u8 = ((image_array_blurred + 10) * 4).astype('uint8')
        thresholded = cv2.adaptiveThreshold(
            image_array_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -4,
        )
        contours_raw, hierarchy = cv2.findContours(
            thresholded,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = (Contour(contour_raw) for contour_raw in contours_raw)
        contours_filtered = sorted(
            (contour for contour in contours if contour.area > 0),
            key=(lambda contour: contour.area),
            reverse=True,
        )
        print(f'Identified {len(contours_filtered)} potential sites')
        if len(contours_filtered) < roi_number:
            warnings.warn(f'Did not find desired number ({roi_number}) of sites.')
        elif len(contours_filtered) > roi_number:
            print(f'Keeping largest {roi_number} sites by area.')
        site_contours = contours_filtered[:roi_number]

        site_rois = [
            ROI.from_center(
                center_x=contour.centroid[0],
                center_y=(self.averaged_image.yshift + contour.centroid[1]),
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
        return self.averaged_image.detect_site_rois(neighborhood_size, detection_threshold, roi_size, restricted_ROI = restricted_ROI)

    def weight_functions(self, rois: Sequence[ROI], background_subtract: bool = False):
        img = self.averaged_image if background_subtract else self.averaged_image.raw_image()
        return [
            img.roi_view(roi) / img.roi_sum(roi) * roi.pixel_area
            for roi in rois
        ]

    @classmethod
    def load_from_h5(cls, h5_path: Union[str, PathLike], use_averaged_background = False, include_2_images = False):
        sequence_dir = Path(h5_path)
        cls.folder = sequence_dir
        shots_h5s = sequence_dir.glob('20*.h5')
        print('Loading imagess...')
        images: list[Image] = []
        for shot in shots_h5s:
            print(shot)
            processor = TweezerPreprocessor(
                load_type='h5',
                h5_path=shot,
                use_averaged_background = use_averaged_background,
                load_rois_threshs=False,
            )
            images.append(processor.images[0])
            if include_2_images:
                images.append(processor.images[1])

        return cls(images)

    def plot_sites(self, rois: Sequence[ROI]):
        '''
        Plot the average of the 1st image taken from each shot
        Include rois
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')

        rois_bbox = ROI.bounding_box(rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)

        fig.suptitle(f'Tweezer site detection ({len(self.images)} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=np.max(self.averaged_image.subtracted_array),
        )

        full_ylims = (
            self.averaged_image.yshift,
            self.averaged_image.yshift + self.averaged_image.height,
        )

        im = self.averaged_image.imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=ax,
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=ax)

        ROI.plot_rois(
            ax,
            rois,
            label_sites=5,
        )

        fig.savefig(f'{self.folder}/tweezers_averaged_image_with_site_rois.pdf')

    def plot_contour_site_detection(self, fig: Figure):
        axs = fig.subplots(nrows=3)
        # print(self.rois)

        rois_bbox = ROI.bounding_box(self.rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)
        full_ylims = (
            self.averaged_image.yshift,
            self.averaged_image.yshift + self.averaged_image.height,
        )
        view_roi = ROI.from_roi_xy(atom_roi_xlims, full_ylims)

        fig.suptitle(f'Tweezer site detection ({len(self.images)} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=np.max(self.averaged_image.subtracted_array),
        )

        im = self.averaged_image.imshow_view(
            view_roi,
            ax=axs[0],
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=axs)

        ROI.plot_rois(
            axs[0],
            self.rois,
            label_sites=5,
        )

        axs[1].set_title('Blurred image')
        # print(self.image_blurred.shape)

        plot_extent = np.array([view_roi.xmin, view_roi.xmax, view_roi.ymax, view_roi.ymin]) - 0.5
        axs[1].imshow(
            self.image_blurred[:, view_roi.xmin:view_roi.xmax],
            extent=plot_extent,
            **raw_img_color_kw,
        )

        axs[2].set_title('Adaptive threshold')
        axs[2].imshow(
            self.image_thresholded[:, view_roi.xmin:view_roi.xmax],
            extent=plot_extent,
            cmap='Greys_r',
        )
        
        ROI.plot_rois(
            axs[2],
            self.rois,
            label_sites=5,
        )

        centroids = np.array([contour.centroid for contour in self.site_contours])

        # print(centroids + [0, self.averaged_image.yshift])
        # print(f'{self.averaged_image.yshift=}')
        
        centroid_x, centroid_y = centroids.T
        axs[2].scatter(
            centroid_x,
            centroid_y + self.averaged_image.yshift,
            s=1,
            color='red',
        )

    def plot_averaged_images(self, rois: Sequence[ROI]):
        '''
        Plot the average of the 1st and 2nd images taken from each shot seperately
        Doesn't include rois for now
        '''
        fig, axs = plt.subplots(nrows=2, ncols=1, layout='constrained')

        rois_bbox = ROI.bounding_box(rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)

        fig.suptitle(f'Tweezer site detection ({len(self.images[::2])} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=np.max(self.averaged_image.subtracted_array),
        )

        full_ylims = (
            self.averaged_image.yshift,
            self.averaged_image.yshift + self.averaged_image.height,
        )

        im = self.avg_1st_img.imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=axs[0],
            **raw_img_color_kw,
        )
        im = self.avg_2nd_img.imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=axs[1],
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=axs)
        fig.savefig(f'{self.folder}/tweezers_averaged_images.pdf')
