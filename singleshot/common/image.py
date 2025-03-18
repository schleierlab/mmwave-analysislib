from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union

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
        return self.subtracted_array[roi.ymin:roi.ymax, roi.xmin:roi.xmax]

    def roi_mean(self, roi: ROI):
        return np.mean(self.roi_view(roi))

    def roi_sum(self, roi: ROI):
        return np.sum(self.roi_view(roi))

    def roi_sums(self, rois: Sequence[ROI]):
        return np.array([self.roi_sum(roi) for roi in rois])


    # @staticmethod
    # def gauss2D(x, amplitude, mux, muy, sigmax, sigmay, rotation, offset):
    #     """
    #     2D Gaussian, see: https://en.wikipedia.org/wiki/Gaussian_function
    #     Parameters:
    #     -----------
    #     x: tuple of 2 arrays, x[0] for x and x[1] for y
    #     amplitude: amplitude of the gaussian, float
    #     mux: center of the gaussian in x, float
    #     muy: center of the gaussian in y, float
    #     sigmax: width of the gaussian in x, float
    #     sigmay: width of the gaussian in y, float
    #     rotation: rotation of the gaussian, float
    #     offset: offset of the gaussian, float

    #     Returns
    #     -------
    #     G: array like shape[1,:],
    #       1d ravel of the gaussian
    #     """
    #     assert len(x) == 2
    #     X = x[0]
    #     Y = x[1]
    #     A = (np.cos(rotation)**2)/(2*sigmax**2) + (np.sin(rotation)**2)/(2*sigmay**2)
    #     B = (np.sin(rotation*2))/(4*sigmay**2) - (np.sin(2*rotation))/(4*sigmax**2)
    #     C = (np.sin(rotation)**2)/(2*sigmax**2) + (np.cos(rotation)**2)/(2*sigmay**2)
    #     G = amplitude * np.exp(-(A * (X - mux) ** 2 + 2 * B * (X - mux) * (Y - muy) + C * (Y - muy) ** 2)) + offset  # + slopex * X + slopey * Y + offset
    #     return G.ravel()  # np.ravel() Return a contiguous flattened array.

    # def get_atom_gaussian_fit(
    #         self,
    #         roi: ROI,
    #         fit_amp_only: bool = False,
    #         gaussian_fit_params = None,
    # ):
    #     """
    #     measure the atom number, temperature of atom through time of flight
    #     the waist of the cloud is determined through 2D gaussian fitting
    #     all fitting parameters are saved in processed_quantities.h5

    #     Parameters
    #     ----------
    #     roi: ROI
    #         The region of interest in which to run the fit.
    #     fit_amp_only: bool
    #         Whether to allow only the amplitude to float during the fit,
    #         versus allowing center, width, rotation, and offset to float.
    #     gaussian_fit_params: array_like, shape [7]
    #     only used when option is "amplitude only", the center, waist, rotation and offset is fixed using the values in gaussian_fit_params

    #     Returns
    #     -------
    #     time_of_flight: float
    #         extracted from global variable "bm_tof_imaging_delay"
    #     atom_number_gaussian: float
    #         atom number through gaussian fitting
    #     guassian_waist: array_like, shape [2]
    #     temperature: array_like, shape [2]
    #         instant temperature assuming initial cloud size is 0
    #         T = m / k_B * (waist/time_of_flight)^2
    #     """
    #     # Creating a grid for the Gaussian fit
    #     [roi_x, roi_y] = self.atoms_roi
    #     x_size = roi_x[1]-roi_x[0]
    #     y_size = roi_y[1]-roi_y[0]
    #     x = np.linspace(0, x_size-1, x_size)
    #     y = np.linspace(0, y_size-1, y_size)
    #     x, y = np.meshgrid(x, y)

    #     # Fitting the 2d Gaussian
    #     initial_guess = np.array([
    #         np.max(self.roi_atoms),
    #         x_size/2,
    #         y_size/2,
    #         x_size/2,
    #         y_size/2,
    #         0,
    #         np.min(self.roi_atoms),
    #     ])

    #     if not fit_amp_only:
    #         popt, pcov = opt.curve_fit(
    #             self.gauss2D,
    #             (y, x),
    #             self.roi_atoms.ravel(),
    #             p0=initial_guess,
    #         )

    #         amplitude, mux, muy, sigmax, sigmay, rotation, offset = popt
    #         gaussian_integral = np.abs(2 * pi * amplitude * sigmax * sigmay)
    #         atom_number_gaussian = gaussian_integral / self.counts_per_atom

    #         sigma = np.sort(np.abs([popt[3], popt[4]]))  # gaussian waiast in pixel, [short axis, long axis]
    #     else:
    #         if gaussian_fit_params is None:
    #             raise ValueError('when choose amplitude only option, '
    #             'you need to put in the gaussain_fit_params')
    #         popt, pcov = opt.curve_fit(
    #             lambda xy, A, offset: self.gauss2D(
    #                 xy, A, *gaussian_fit_params[1:6], offset
    #             ),
    #             (y, x),
    #             self.roi_atoms.ravel(),
    #             p0=(self.roi_atoms.max(), 0),
    #         )
    #         atom_number_gaussian = np.abs(2 * pi * (popt[0]-popt[1]) * gaussian_fit_params[3] * gaussian_fit_params[4] / self.counts_per_atom)
    #         sigma = np.sort(np.abs([gaussian_fit_params[3], gaussian_fit_params[4]]))  # gaussian waiast in pixel, [short axis, long axis]

    #     gaussian_waist = np.array(sigma)*self.imaging_setup.atom_plane_pixel_size  # convert from pixel to distance m
    #     time_of_flight = self.globals['bm_tof_imaging_delay']

    #     temperature = self.get_atom_temperature(time_of_flight, gaussian_waist)

    #     self.save_atom_temperature(atom_number_gaussian, time_of_flight, gaussian_waist, temperature)

    #     return time_of_flight, atom_number_gaussian, gaussian_waist, temperature
