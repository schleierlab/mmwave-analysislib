from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union, cast
from typing_extensions import Unpack

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
from numpy.typing import NDArray
from scipy import ndimage, optimize

from analysislib.common.plot_config import PlotConfig
from analysislib.common.typing import Quadruple, RectangleKwargs


class ROI(NamedTuple):
    '''
    Barebones class to wrap the four numbers needed to specify a ROI (region of interest) in an image.
    Parameters xmin and ymin are inclusive, xmax and ymax are exclusive, and are
    all given in units of pixels. Thus, the center pixel of ROI(xmin=13, xmax=18, ymin=25, ymax=30),
    a 5px x 5px region, is (x, y) = (15, 27).
    '''
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def pixel_area(self):
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        return (self.xmax + self.xmin - 1) / 2, (self.ymax + self.ymin - 1) / 2

    def patch(self, scale_factor: float = 1.0, **kwargs: Unpack[RectangleKwargs]):
        default_kw = RectangleKwargs(linewidth=0.75, edgecolor='r', facecolor='none')
        return patches.Rectangle(
            ((self.xmin - 0.5) * scale_factor, (self.ymin - 0.5) * scale_factor),
            self.width * scale_factor, self.height * scale_factor,
            **(default_kw | kwargs),
        )
    
    def contains(self, x, y) -> bool:
        return (self.xmin <= x < self.xmax) and (self.ymin <= y < self.ymax)

    @staticmethod
    def bounding_box(rois: Sequence[ROI]) -> ROI:
        return ROI(
            min(roi.xmin for roi in rois),
            max(roi.xmax for roi in rois),
            min(roi.ymin for roi in rois),
            max(roi.ymax for roi in rois),
        )

    @classmethod
    def from_center(cls, center_x: float, center_y: float, size: int) -> ROI:
        if size < 1:
            raise ValueError('size must be positive integer')
        halfsize = size / 2
        # the extra +0.5 terms make sure that center pixel is in the middle
        # when we use the ROI bounds as a slice(inclusive left endpt, exclusive right endpt),
        # as we normally do for tweezers
        return cls(
            int(center_x - halfsize + 0.5),
            int(center_x + halfsize + 0.5),
            int(center_y - halfsize + 0.5),
            int(center_y + halfsize + 0.5),
        )

    @classmethod
    def from_roi_xy(cls, roi_x: tuple[int, int], roi_y: tuple[int, int]):
        return cls(roi_x[0], roi_x[1], roi_y[0], roi_y[1])

    @staticmethod
    def toarray(rois: Sequence[ROI]):
        return np.array([roi for roi in rois])

    @staticmethod
    def fromarray(arr) -> list[ROI]:
        return [ROI(roi[0], roi[1], roi[2], roi[3]) for roi in arr]

    @staticmethod
    def plot_rois(
        ax: Axes,
        rois: Sequence[ROI],
        edgecolors: Optional[Sequence[ColorType]] = None,
        label_sites: Optional[int] = 5,
        label_displacement: tuple[float, float] = (0, -8),
        plot_config: Optional[PlotConfig] = None,
    ):
        plotconfig = plot_config or PlotConfig()
        edgecolor_iter = itertools.repeat('yellow') if edgecolors is None else edgecolors
        patches: tuple[Rectangle, ...] = tuple(
            roi.patch(edgecolor=edgecolor, alpha=0.6)
            for roi, edgecolor in zip(rois, edgecolor_iter)
        )
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

        if label_sites is None:
            return
        if label_sites <= 0:
            raise ValueError
        for j, roi in enumerate(rois):
            if j % label_sites == 0:
                ax.annotate(
                    str(j), # The site index to display
                    xy=(roi.center[0] + label_displacement[0], roi.center[1] + label_displacement[1]), # Position of the text
                    horizontalalignment='center',
                    **plotconfig.tweezer_index_label_kw,
                )


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
        return np.broadcast_to(self.background, self.array.shape).astype(self.array.dtype)

    @property
    def subtracted_array(self):
        return self.array - self.bkg_array

    def raw_image(self) -> Image:
        return Image(self.array, background=0, yshift=self.yshift)

    def background_image(self) -> Image:
        return Image(self.bkg_array, background=0, yshift=self.yshift)

    @classmethod
    def mean(cls, images: Sequence[Image]) -> Image:
        yshift = images[0].yshift
        if any(image.yshift != yshift for image in images[1:]):
            raise ValueError('All images must have the same yshift.')

        return Image(
            np.mean([image.array for image in images], axis=0),
            cls.mean_background(images),
            yshift,
        )

    @staticmethod
    def mean_background(images: Sequence[Image]) -> NDArray:
        # computed manually to allow for broadcasted backgrounds
        # use larger ints to avoid overflow
        bkg_generator = (np.asarray(image.background).astype(np.int32) for image in images)
        return sum(bkg_generator, start=np.array(0)) / len(images)  # need np.array in start to guarantee NDArray output

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

    def imshow_view(
            self,
            roi: Optional[ROI] = None,
            scale_factor: float = 1.0,
            ax: Optional[Axes] = None,
            **kwargs,
    ):
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
        if roi is None:
            im = ax.imshow(
                self.subtracted_array,
            **kwargs,
            )
        else:
            extent = cast(Quadruple, tuple(
                scale_factor * (value - 0.5)
                for value in [roi.xmin, roi.xmax, roi.ymax, roi.ymin]
            ))
            im = ax.imshow(
                self.roi_view(roi),
                extent=extent,
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

    def detect_site_rois(
            self,
            neighborhood_size: int,
            detection_threshold: float,
            roi_size: int,
            restricted_ROI = None,
    ):
        from scipy import ndimage

        if restricted_ROI is None:
            data = self.subtracted_array
            x_shift = 0
            y_shift = self.yshift
        else:
            data = self.roi_view(restricted_ROI)
            x_shift = restricted_ROI.xmin
            y_shift = restricted_ROI.ymin

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

        site_rois: list[ROI] = []
        for dy, dx in slices:
            center_x = x_shift + (dx.start + dx.stop) / 2
            center_y = y_shift + (dy.start + dy.stop) / 2

            site_rois.append(ROI.from_center(center_x, center_y, roi_size))

        # Sort rois by x1 coordinate
        # We want to sort the site rois from x large to x small, because x large corresponds to site 0 for rearrangement
        # (which is also the site with the smallest frquency)
        site_rois.sort(key=lambda roi: roi.xmin, reverse=True) # range from from big to small

        return site_rois

    def roi_fit_gaussian2d(
            self,
            roi: ROI,
            isotropic: bool = False,
            small_dot: bool = False,
            blur: Optional[int] = None,
    ):
        """
        Fits a 2D Gaussian function to the image data.
        Mainly intended to fit images of the MOT.

        Parameters
        ----------
        roi: ROI
            Area to fit.
        isotropic: bool
            If true, constrain the Gaussian to be rotationally symmetric.
            Otherwise, Gaussian is only constrained to have principal axes
            along Cartesian axes.
        blur: int, optional
            If specified, blur the image with a Gaussian filter of
            the specified width (in pixels).
        
        Returns
        -------
        popt, pcov: arrays
            Optimal fit parameters and the estimated covariance of popt.
            Parameter order:
            If isotropic:
                x0, y0, width, peak_height, offset
            else:
                x0, y0, width, height, peak_height, offset
        """
        if blur is not None:
            roiview = ndimage.gaussian_filter(
                self.roi_view(roi),
                sigma=(blur, blur),
                order=0,
            )
        else:
            roiview = self.roi_view(roi)

        # y, x = np.mgrid[:roiview.shape[0], :roiview.shape[1]]
        y, x = np.mgrid[
            roi.ymin:roi.ymax,
            roi.xmin:roi.xmax,
        ]
        xys = np.vstack([x.ravel(), y.ravel()]).T

        x0_guess_rel, y0_guess_rel = np.unravel_index(np.argmax(roiview), roiview.shape)
        x0_guess, y0_guess = x0_guess_rel + roi.xmin, y0_guess_rel + roi.ymin
        width_guess = roi.width/4
        height_guess = roi.height/4

        if small_dot:
            width_guess = width_guess/8
            height_guess = height_guess/8

        z_data_range = np.max(roiview) - np.min(roiview)
        a_guess = z_data_range
        offset_guess = np.min(roiview)

        if isotropic:
            p0 = [x0_guess, y0_guess, width_guess, a_guess, offset_guess]
            return optimize.curve_fit(
                self.gaussian2d_uniform,
                xys,
                roiview.ravel(),
                p0=p0,
                bounds=np.array([
                    (roi.xmin, roi.xmax),
                    (roi.ymin, roi.ymax),
                    (0, roi.width),
                    (0, np.inf),
                    (-np.inf, np.inf),
                ]).T,
            )
        else:
            p0 = [x0_guess, y0_guess, width_guess, height_guess, a_guess, offset_guess]
            return optimize.curve_fit(
                lambda xy, x0, y0, width, height, peak_height, offset: self.gaussian2d(
                    xy,
                    x0,
                    y0,
                    width,
                    height,
                    rotation=0,
                    peak_height=peak_height,
                    offset=offset,
                ),
                xys,
                roiview.ravel(),
                p0=p0,
                bounds=np.array([
                    (roi.xmin, roi.xmax),
                    (roi.ymin, roi.ymax),
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
        z_score_vector = centered_xy / [width, width]

        # shape (...,)
        mahalanobis_dist_sq = np.sum(z_score_vector**2, axis=-1)

        return offset + peak_height * np.exp(-mahalanobis_dist_sq / 2)
