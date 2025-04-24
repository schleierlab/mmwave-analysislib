from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union, cast

import numpy as np
from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.constants import pi

MaybeInt = Optional[int]

class ROI(NamedTuple):
    '''
    Barebones class to wrap the four numbers needed to specify a ROI (region of interest) in an image.
    xmin and ymin are inclusive, xmax and ymax are exclusive.
    '''
    xmin: MaybeInt
    xmax: MaybeInt
    ymin: MaybeInt
    ymax: MaybeInt

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def pixel_area(self):
        return self.width * self.height

    def patch(self, scale_factor: float = 1.0, **kwargs):
        default_kw = dict(linewidth=0.75, edgecolor='r', facecolor='none')
        return patches.Rectangle(
            (self.xmin * scale_factor, self.ymin * scale_factor),
            self.width * scale_factor, self.height * scale_factor,
            **(default_kw | kwargs),
        )

    def bounding_box(rois: Sequence[ROI]):
        return ROI(
            min(roi.xmin for roi in rois),
            max(roi.xmax for roi in rois),
            min(roi.ymin for roi in rois),
            max(roi.ymax for roi in rois),
        )

    @classmethod
    def from_roi_xy(cls, roi_x: tuple[MaybeInt, MaybeInt], roi_y: tuple[MaybeInt, MaybeInt]):
        return cls(roi_x[0], roi_x[1], roi_y[0], roi_y[1])

    @staticmethod
    def toarray(rois: Sequence[ROI]):
        return np.array([roi for roi in rois])

    @staticmethod
    def fromarray(arr: ArrayLike) -> list[ROI]:
        return [ROI(roi[0], roi[1], roi[2], roi[3]) for roi in arr]


@dataclass
class Image:
    '''
    Barebones class to wrap a 2D image array and a background array,
    together with convenience functions for cropping, averaging, etc.
    '''
    array: NDArray
    background: Union[NDArray, Literal[0]] = 0

    yshift: int = 0
    '''
    Pixels to shift vertically in the image array for indexing purposes.
    A ROI with ylims [30, 40] for an image with yshift = 10 would give rows
    with indices 20, 21, ..., 29.
    '''

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def bkg_array(self):
        return np.broadcast_to(self.background, self.array.shape)

    @property
    def subtracted_array(self):
        return self.array - self.bkg_array

    def raw_image(self) -> Image:
        return Image(self.array, background=0, yshift=self.yshift)

    def background_image(self) -> Image:
        return Image(self.bkg_array, background=0, yshift=self.yshift)

    @staticmethod
    def mean(images: Sequence[Image]) -> Image:
        # computed manually to allow for broadcasted backgrounds
        # use larger ints to avoid overflow
        background = sum(image.background.astype(np.int32) for image in images) / len(images)

        yshift = images[0].yshift
        if any(image.yshift != yshift for image in images[1:]):
            raise ValueError('All images must have the same yshift.')

        return Image(
            np.mean([image.array for image in images], axis=0),
            background,
            yshift,
        )

    def roi_view(self, roi: ROI):
        '''
        Returns a view of the full image cropped to the specified ROI.
        '''
        yslice = slice(
            None if roi.ymin is None else roi.ymin - self.yshift,
            None if roi.ymax is None else roi.ymax - self.yshift
        )
        return self.subtracted_array[
            yslice,
            roi.xmin:roi.xmax,
        ]

    def imshow_view(self, roi: ROI, scale_factor: float = 1.0, ax: Optional[Axes] = None, **kwargs):
        '''
        Parameters
        ----------
        roi : ROI
            Region of interest to display
        ax : Axes, optional
            Axis to plot on. If not provided, a new figure will be created.
        kwargs
            Additional keyword arguments to pass to imshow
        '''
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(
            self.roi_view(roi),
            extent=(scale_factor * np.array([roi.xmin, roi.xmax, roi.ymax, roi.ymin])),
            **kwargs,
        )
        return im

    def roi_mean(self, roi: ROI):
        return np.mean(self.roi_view(roi))

    def roi_stddev(self, roi: ROI):
        return np.std(self.roi_view(roi))

    def roi_sum(self, roi: ROI):
        return np.sum(self.roi_view(roi))

    def roi_sums(self, rois: Sequence[ROI]):
        return np.array([self.roi_sum(roi) for roi in rois])

    def detect_site_rois(self, neighborhood_size: int, detection_threshold: float, roi_size: int):
        from scipy import ndimage

        data = self.subtracted_array
        data_maxfilt = ndimage.maximum_filter(data, neighborhood_size)
        data_minfilt = ndimage.minimum_filter(data, neighborhood_size)

        local_maxima = (data == data_maxfilt)
        contrast_filt = ((data_maxfilt - data_minfilt) > detection_threshold)
        prominent_maxima = np.logical_and(
            local_maxima,
            contrast_filt,
        )

        labeled, _ = ndimage.label(prominent_maxima)
        slices = cast(
            list[tuple[slice, slice]],
            ndimage.find_objects(labeled),
        )

        rois: list[ROI] = []
        halfsize = roi_size / 2
        for dy, dx in slices:
            center_x = (dx.start + dx.stop) / 2
            center_y = self.yshift + (dy.start + dy.stop) / 2

            rois.append(ROI(
                int(center_x - halfsize),
                int(center_x + halfsize),
                int(center_y - halfsize),
                int(center_y + halfsize),
            ))
        # Sort rois by x1 coordinate
        rois.sort(key=lambda roi: roi.xmin)

        return rois

    def roi_fit_gaussian2d(self, roi: ROI, uniform = False):
        """
        Fits a 2D Gaussian function to the image data.
        Mainly intended to fit images of the MOT.
        """
        roiview = self.roi_view(roi)
        y, x = np.mgrid[:roiview.shape[0], :roiview.shape[1]]
        xys = np.vstack([x.ravel(), y.ravel()]).T

        x0_guess, y0_guess = np.unravel_index(np.argmax(roiview), roiview.shape)
        width_guess = roi.width/4
        height_guess = roi.height/4
        z_data_range = np.max(roiview) - np.min(roiview)
        a_guess = z_data_range
        offset_guess = np.min(roiview)

        if uniform:
            p0 = [x0_guess, y0_guess, width_guess, a_guess, offset_guess]
            return optimize.curve_fit(
                lambda xy, x0, y0, width, peak_height, offset :self.gaussian2d_uniform(xy, x0, y0, width, peak_height, offset),
                xys,
                roiview.ravel(),
                p0=p0,
                bounds=np.array([
                    (0, roi.width),
                    (0, roi.height),
                    (0, roi.width),
                    (0, np.inf),
                    (-np.inf, np.inf),
                ]).T,
            )
        else:
            p0 = [x0_guess, y0_guess, width_guess, height_guess, a_guess, offset_guess]
            return optimize.curve_fit(
                lambda xy, x0, y0, width, height, peak_height, offset :self.gaussian2d(xy, x0, y0, width, height, rotation = 0, peak_height = peak_height, offset = offset),
                xys,
                roiview.ravel(),
                p0=p0,
                bounds=np.array([
                    (0, roi.width),
                    (0, roi.height),
                    (0, roi.width),
                    (0, roi.height),
                    (0, np.inf),
                    (-np.inf, np.inf),
                ]).T,
            )

    @staticmethod
    def gaussian2d(xy, x0, y0, width, height, rotation, peak_height, offset):
        """
        Returns an axis-parallel 2D Gaussian function with constant offset.

        Parameters
        ----------
        xy : array_like, shape (..., 2)
            Input coordinates as (x, y) arrays
        x0, y0 : float
            Center position of the Gaussian
        width, height : float
            Width parameters of the Gaussian in x and y directions
        rotation : float
            Rotation angle of the first principal axes of the Gaussian, in radians,
            relative to the x-axis.
        a : float
            Peak height of the Gaussian, without the offset
        offset : float
            Offset of the Gaussian

        Returns
        -------
        array, shape (...,)
            Gaussian function values
        """
        inverse_rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation)], [-np.sin(rotation), np.cos(rotation)]])
        centered_xy = np.asarray(xy) - np.asarray([x0, y0])  # shape (..., 2)

        # shape (..., 2)
        z_score_vector = np.dot(centered_xy, inverse_rotation_matrix.T) / [width, height]

        # shape (...,)
        mahalanobis_dist_sq = np.sum(z_score_vector**2, axis=-1)

        return offset + peak_height * np.exp(-mahalanobis_dist_sq / 2)

    @staticmethod
    def gaussian2d_uniform(xy, x0, y0, width, peak_height, offset):
        """
        Returns an axis-parallel 2D Gaussian function with constant offset.

        Parameters
        ----------
        xy : array_like, shape (..., 2)
            Input coordinates as (x, y) arrays
        x0, y0 : float
            Center position of the Gaussian
        width : float
            Width parameters of the Gaussian in x and y directions. Assume they are the same
        peak_height : float
            Peak height of the Gaussian, without the offset
        offset : float
            Offset of the Gaussian

        Returns
        -------
        array, shape (...,)
            Gaussian function values
        """
        centered_xy = np.asarray(xy) - np.asarray([x0, y0])  # shape (..., 2)

        # shape (..., 2)
        z_score_vector = centered_xy/ [width, width]

        # shape (...,)
        mahalanobis_dist_sq = np.sum(z_score_vector**2, axis=-1)

        return offset + peak_height * np.exp(-mahalanobis_dist_sq / 2)
