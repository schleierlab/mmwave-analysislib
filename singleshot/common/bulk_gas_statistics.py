from typing import Optional

import h5py
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy import optimize
import uncertainties
from pathlib import Path
import lyse # needed for MLOOP
from analysislib.analysis.data import h5lyze as hz # needed for testing MLOOP

from .plot_config import PlotConfig


globals_friendly_names = {
    'mw_detuning': 'Microwave detuning',
}


class BulkGasStatistician:
    """Class for statistical analysis of tweezer imaging data.

    This class provides methods for statistical analysis of tweezer imaging data.
    It also generates several different types of plots for visualizing the data and
    manages input to MLOOP for online optimization.

    Parameters
    ----------
    preproc_h5_path : str
        Path to the processed quantities h5 file
    shot_h5_path : str
        Path to the shot h5 file, we only need this for MLOOP to save results for optimization
    plot_config : PlotConfig, optional
        Configuration object for plot styling
    """
    def __init__(self, 
                 preproc_h5_path: str, 
                 shot_h5_path: str, 
                 plot_config: PlotConfig = None
                 ):
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(preproc_h5_path)
        self._save_mloop_params(shot_h5_path)
        h5p = Path(preproc_h5_path)
        self.folder_path = h5p.parent

    def _load_processed_quantities(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.atom_numbers = f['atom_numbers'][:]
            self.params_list = f['params'][:]
            self.n_runs = f.attrs['n_runs']
            self.current_params = f['current_params'][:]
            if 'gaussian_cloud_params' in f:
                self.gaussian_cloud_params = f['gaussian_cloud_params'][:]

    def _save_mloop_params(self, shot_h5_path: str) -> None:
        """Save values and uncertainties to be used by MLOOP for optimization.

        MLOOP reads the results of any experiment from the latest shot h5 file, 
        updates the loss landscape, and triggers run manager for the next batch
        of experiments.
        
        Parameters
        ----------
        shot_h5_path : str
            Path to the shot h5 file
        """
        # Save values for MLOOP
        # Save sequence analysis result in latest run
        run = lyse.Run(h5_path=shot_h5_path)
        my_condition = True
        # run.save_result(name='survival_rate', value=survival_rate if my_condition else np.nan)
        # with h5py.File(shot_h5_path, mode='r+') as f:
        #     globals = f['globals']
        #     global_var = hz.getAttributeDict(globals['Diagnostics'])['x']
        #     global_var = eval(global_var)
        survival_rate = 0#np.exp(-global_var**2/5)
        survival_uncertainty = 0.1
        mloop_result = (survival_rate, survival_uncertainty)
        run.save_results_dict(
            {
                'survival_rate': mloop_result if my_condition else (np.nan, np.nan),
            },
            uncertainties=True,
        )

    def plot_atom_number(self, fig: Optional[Figure] = None, plot_lorentz = True):
        """Plot atom number vs the looped parameter (simplest is shot number) and save
        the image in the folder path.

        This function requires that get_atom_number has been run first to save the
        atom number to processed_quantities.h5.

        Parameters
        ----------
        fig : Optional[Figure], default=None
            Figure to plot on, if None a new figure is created
        plot_lorentz : bool, default=True
            Whether to plot a Lorentzian fit to the atom number data
        """
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
            is_subfig = False
        else:
            ax = fig.subplots()
            is_subfig = True

        loop_params = np.squeeze(self.current_params)

        # Group data points by x-value and calculate statistics
        unique_params = np.unique(loop_params)
        means = np.array([
            np.mean(self.atom_numbers[loop_params == x])
            for x in unique_params
        ])
        stds = np.array([
            np.std(self.atom_numbers[loop_params == x])
            for x in unique_params
        ])

        ax.errorbar(
            unique_params,
            means,
            yerr=stds,
            marker='.',
            linestyle='-',
            alpha=0.5,
            capsize=3,
        )
        ax.set_xlabel(
            f"{self.params_list[0][0].decode('utf-8')} [{self.params_list[0][1].decode('utf-8')}]",
            fontsize=self.plot_config.label_font_size,
        )
        ax.set_ylabel(
            'atom count',
            fontsize=self.plot_config.label_font_size,
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.plot_config.label_font_size,
        )

        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')

        # doing the fit at the end of the run
        if self.atom_numbers.shape[0] == self.n_runs and plot_lorentz:
            popt, pcov = self.fit_lorentzian(unique_params, means)
            upopt = uncertainties.correlated_values(popt, pcov)

            x_plot = np.linspace(
                np.min(unique_params),
                np.max(unique_params),
                1000,
            )

            ax.plot(x_plot, self.lorentzian(x_plot, *popt))
            fig.suptitle(
                f'Center frequency: ${upopt[0]:SL}$ MHz; '
                f'Width: ${1e+3 * upopt[1]:SL}$ kHz'
            )

        figname = f"{self.folder_path}\count vs param.png"
        if is_subfig:
            self.save_subfig(fig, figname)
        else:
            fig.savefig(figname)

    def plot_mot_params(self, fig: Optional[Figure] = None, show_means=True):
        """Plot atom number vs the looped parameter and save the image in the folder path.

        This function requires that get_atom_number has been run first to save the
        atom number to processed_quantities.h5.

        Parameters
        ----------
        fig : Optional[Figure], default=None
            Figure to plot on, if None a new figure is created
        plot_lorentz : bool, default=True
            Whether to plot a Lorentzian fit to the atom number data
        """
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
            is_subfig = False
        else:
            ax = fig.subplots()
            is_subfig = True

        loop_params = np.squeeze(self.current_params)

        # Group data points by x-value and calculate statistics
        unique_params = np.unique(loop_params)
        
        means = np.array([
            np.mean(self.gaussian_cloud_params[loop_params == x])
            for x in unique_params
        ])
        stds = np.array([
            np.std(self.gaussian_cloud_params[loop_params == x])
            for x in unique_params
        ])

        ax.errorbar(
            unique_params,
            means,
            yerr=stds,
            marker='.',
            linestyle='-',
            alpha=0.5,
            capsize=3,
        )
        ax.set_xlabel(
            f"{self.params_list[0][0].decode('utf-8')} [{self.params_list[0][1].decode('utf-8')}]",
            fontsize=self.plot_config.label_font_size,
        )
        ax.set_ylabel(
            'atom count',
            fontsize=self.plot_config.label_font_size,
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.plot_config.label_font_size,
        )

        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')

        # doing the fit at the end of the run
        if self.atom_numbers.shape[0] == self.n_runs and plot_lorentz:
            popt, pcov = self.fit_lorentzian(unique_params, means)
            upopt = uncertainties.correlated_values(popt, pcov)

            x_plot = np.linspace(
                np.min(unique_params),
                np.max(unique_params),
                1000,
            )

            ax.plot(x_plot, self.lorentzian(x_plot, *popt))
            fig.suptitle(
                f'Center frequency: ${upopt[0]:SL}$ MHz; '
                f'Width: ${1e+3 * upopt[1]:SL}$ kHz'
            )

        figname = f"{self.folder_path}\count vs param.png"
        if is_subfig:
            self.save_subfig(fig, figname)
        else:
            fig.savefig(figname)


    @staticmethod
    def save_subfig(subfig, filename):
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

    def fit_lorentzian(self, x_data, y_data):
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
        return optimize.curve_fit(self.lorentzian, x_data, y_data, p0=p0)