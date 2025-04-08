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
        x0_guess = x_data[np.argmax(y_data)]
        x_resolution = (x_data[1] - x_data[0])
        width_guess = 2*x_resolution
        y_data_range = np.max(y_data) - np.min(y_data)
        a_guess = y_data_range / width_guess * (width_guess/2)**2

        offset_guess = np.min(y_data)
        p0 = [x0_guess, width_guess, a_guess, offset_guess]
        return optimize.curve_fit(self.lorentzian, x_data, y_data, p0=p0, sigma=sigma)