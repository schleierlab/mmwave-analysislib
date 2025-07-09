from abc import abstractmethod, ABC
from os import PathLike
from typing import Optional
from matplotlib.figure import Figure
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# try:
#     lyse
# except NameError:
#     import lyse # needed for MLOOP

from .plot_config import PlotConfig
from .image import ROI
from typing import Union

class BaseStatistician(ABC):
    """Base class for statistical analysis of tweezer or bulk gas imaging data."""
    def __init__(self):
        # TODO: move common init tasks here from child classes
        pass

    @abstractmethod
    def _load_processed_quantities(self, preproc_h5_path: str) -> None:
        """Load processed quantities from an h5 file."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _save_mloop_params(self, shot_h5_path: str) -> None:
        """Save values and uncertainties to be used by MLOOP for optimization."""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def save_subfig(subfig, filename: Union[str, PathLike]):
        # Create a new figure with the same size as the subfigure
        new_fig = plt.figure(figsize=subfig.get_figure().get_size_inches())

        # Get the position and size of the subfigure
        bbox = subfig.get_tightbbox(subfig.figure.canvas.get_renderer())
        bbox = bbox.transformed(subfig.figure.transFigure.inverted())

        # Create a new axes that fills the entire figure
        new_axes = new_fig.add_axes([0, 0, 1, 1])

        # Draw the subfigure content
        new_axes.set_facecolor(subfig.get_facecolor())
        subfig.figure.canvas.draw()

        # Save the new figure
        new_fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(new_fig)


    # TODO: maybe we can keep all of our fitting functions here, so that both child classes
    # have access to them and we keep fitting functionality in one place.

    @staticmethod
    # Define the damped Rabi oscillation model
    def rabi_model(t, A, Omega, phi, T2, C, exp_decay = False):
        if exp_decay:
            return A * np.cos(Omega * t + phi) * np.exp(-t / T2) + C
        else:
            return A * np.cos(Omega * t + phi) * np.exp(-(t / T2)**2) + C # gaussian decay

    @staticmethod
    def lorentzian(x, x0, width, a, offset):
        """
        Returns a Lorentzian function.

        Parameters
        ----------
        x : float or array
            Input values
        x0 : float
            Central frequency
        width : float
            Width of the Lorentzian
        a : float
            Amplitude of the Lorentzian
        offset : float
            Offset of the Lorentzian

        Returns
        -------
        float or array
            Lorentzian function values
        """
        detuning = x - x0
        return a * width / ((width / 2)**2 + detuning**2) + offset

    def fit_lorentzian(self, x_data, y_data, sigma=None):
        """
        Fits a Lorentzian function to the atom number data.
        """
        x0_guess = x_data[np.argmin(y_data)]
        x_resolution = (x_data[1] - x_data[0])
        width_guess = 2*x_resolution
        y_data_range = np.max(y_data) - np.min(y_data)
        a_guess = y_data_range / width_guess * (width_guess/2)**2

        offset_guess = np.min(y_data)
        p0 = [x0_guess, width_guess, a_guess, offset_guess]
        return optimize.curve_fit(self.lorentzian, x_data, y_data, p0=p0, sigma=sigma)

    @staticmethod
    def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, theta, offset):
        """
        2D Gaussian function with elliptical shape and rotation.
        """
        x, y = coords
        x_shifted = np.asarray(x) - x0
        y_shifted = np.asarray(y) - y0

        xo = float(x0)
        yo = float(y0)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + A * np.exp(-(a*(x_shifted**2) + 2*b*x_shifted*y_shifted + c*(y_shifted**2)))

        return g
        # return g.ravel()  # probably do not need to do shaping shenanigans

    def fit_gaussian_2d(self, X, Y, data, sigma=None, initial_params=None):
        """
        Fits a 2D Gaussian function to the atom number data.
        """
        X = np.ravel(X)
        Y = np.ravel(Y)
        data = np.ravel(data)

        p0 = initial_params
        if initial_params is None:
            p0 = [np.max(data), X[np.argmax(data)], Y[np.argmax(data)], 1, 1, 0, np.min(data)]

        bounds = (
            [-np.inf, X.min(), Y.min(), 1e-5, 1e-5, -np.pi, -np.inf],  # Lower bounds
            [np.inf, X.max(), Y.max(), X.max(), Y.max(), np.pi, np.inf]  # Upper bounds
        )
        popt, pcov = optimize.curve_fit(
            self.gaussian_2d,
            (X, Y),
            data,
            p0=p0,
            sigma=np.ravel(sigma),
            bounds=bounds,
        )
        return popt, pcov

    def reshape_to_unique_params_dim(self, data, x_params, y_params):
        """
        Reshape the data to have the same shape as the unique parameter combinations.

        Parameters
        ----------
        data : array_like
            The data to reshape.
        x_params : array_like
            The unique values along the x-axis.
        y_params : array_like
            The unique values along the y-axis.

        Returns
        -------
        data_new : array_like
            The reshaped data with x dime as column and y dimension as row.
        """
        # pad with nan if not enough data
        data_new = np.pad(
            data,
            (0, x_params.shape[0]*y_params.shape[0] - data.shape[0]),
            'constant',
            constant_values=np.nan,
        )

        data_new = data_new.reshape([
            y_params.shape[0],
            x_params.shape[0]]
            ) # reshape

        return data_new

    def get_params_order(self, unique_params):
        """
        Function to find which parameters are the first to scan,
        only works for 2 parameters for now

        Parameters
        ----------
        unique_params : array_like
            The unique parameter combinations.

        Returns
        -------
        x_params_index : int
            The index of the first (i.e. *innermost* scanned) parameter.
        y_params_index : int
            The index of the second parameter.

        """

        if unique_params.shape[0] >= 2: # find difference when more than 2 shots in presence
            params_dif = unique_params[1]- unique_params[0]
            x_params_index = np.where(params_dif != 0)[0][0] # find index of difference
            y_params_index = np.delete(np.arange(2), x_params_index).item() # find the other index
        else: # default value with only 1 shot
            x_params_index = 0
            y_params_index = 1

        return x_params_index, y_params_index


    def get_unique_params_along_axis(self, unique_params, index):
        """
        find unique values along different axis

        Parameters
        ----------
        unique_params : array_like
            The unique parameter combinations.
        index : int
            The axis to find unique values along.

        Returns
        -------
        params : array_like
            The unique values along the specified axis.
        """
        params = np.unique(unique_params[:, index]).ravel()

        return params