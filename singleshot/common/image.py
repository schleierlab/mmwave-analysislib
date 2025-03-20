from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union

from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    def bkg_array(self):
        return np.broadcast_to(self.background, self.array.shape)

    @property
    def subtracted_array(self):
        return self.array - self.bkg_array

    def roi_view(self, roi: ROI):
        '''
        Returns a view of the full image cropped to the specified ROI.
        '''
        return self.subtracted_array[
            roi.ymin - self.yshift : roi.ymax - self.yshift,
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

    def roi_fit_gaussian2d(self, roi: ROI):
        """
        Fits a 2D Gaussian function to the image data.
        Mainly intended to fit images of the MOT.
        """
        image = self.roi_view(roi)
        x0_guess, y0_guess = np.unravel_index(np.argmax(image), image.shape)
        width_guess = roi.width*10
        height_guess = roi.height*10
        z_data_range = np.max(image) - np.min(image)
        a_guess = z_data_range / (width_guess * height_guess)

        offset_guess = np.min(image)
        p0 = [x0_guess, y0_guess, width_guess, height_guess, a_guess, offset_guess]
        return optimize.curve_fit(self.gaussian2d, (np.arange(roi.width), np.arange(roi.height)), image, p0=p0)

    @staticmethod
    def gaussian2d(inputs, x0, y0, width, height, a, offset):
        """
        Returns a 2D Gaussian function.

        Parameters
        ----------
        inputs : tuple of float or array
            Input values (x, y)
        x0 : float
            Central frequency
        width : float
            Width of the Gaussian
        a : float
            Amplitude of the Gaussian
        offset : float
            Offset of the Gaussian

        Returns
        -------
        float or array
            Gaussian function values
        """
        x, y = inputs
        return a * np.exp(-((x - x0) / width)**2 - ((y - y0) / height)**2) + offset
