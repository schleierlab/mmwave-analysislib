from typing import Optional

import h5py
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy import optimize
import uncertainties
import uncertainties.unumpy as unumpy
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
            if 'gaussian_cloud_params_nom' in f:
                self.gaussian_cloud_params_nom = f['gaussian_cloud_params_nom'][:]
                self.gaussian_cloud_params_std = f['gaussian_cloud_params_std'][:]

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

        # loop_params = np.squeeze(self.current_params)
        loop_params = self.current_params[:, 0]

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

        ax.set_ylim(bottom=0)

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
        show_means : bool, default=True
            Whether to show the means of the mot params vs loop params as a horizontal line on
            on the plot.
        """
        loop_params = self.current_params[:, 0]
        unique_params = np.unique(loop_params)

        # Create subplots
        if fig is None:
            fig = plt.figure(
                figsize=(12, 8),
                constrained_layout=True
            )
            is_subfig = False
        else:
            axs = fig.subplots(2, 3)
            axs = axs.flatten()
            is_subfig = True

        param_names = ['x₀', 'y₀', 'width', 'height', 'amplitude', 'offset']

        # Convert nominal values and uncertainties to ufloat arrays
        gaussian_cloud_params = unumpy.uarray(
            self.gaussian_cloud_params_nom,
            self.gaussian_cloud_params_std
        )
        #TODO: replace saving the diag(cov) in the h5 with storing the full covariance matrix
        #gaussian_cloud_params = uncertainties.correlated_values(mean_vals, covariance_matrix)


        # For each parameter
        for param_idx, (ax, param_name) in enumerate(zip(axs, param_names)):
            # For each unique x value, compute mean and combined uncertainty
            means = []
            uncertainties = []
            for x in unique_params:
                # Get all measurements for this x value
                mask = (loop_params == x)
                # Get all values for this parameter at the matching x positions
                param_values = gaussian_cloud_params[mask][:, param_idx]

                if len(param_values) > 1:
                    # Multiple measurements: combine fit uncertainties with shot-to-shot variance
                    mean_value = np.mean(unumpy.nominal_values(param_values))
                    shot_to_shot_var = np.var(unumpy.nominal_values(param_values))
                    # Average the individual fit variances
                    fit_var = np.mean(unumpy.std_devs(param_values)**2)
                    # Total uncertainty is sqrt of sum of variances
                    total_uncertainty = np.sqrt(shot_to_shot_var + fit_var)
                else:
                    # Single measurement: just use the fit uncertainty
                    mean_value = unumpy.nominal_values(param_values[0])
                    total_uncertainty = unumpy.std_devs(param_values[0])

                means.append(mean_value)
                uncertainties.append(total_uncertainty)

            means = np.array(means)
            uncertainties = np.array(uncertainties)

            # Plot with error bars
            ax.errorbar(
                unique_params,
                means,  # No need for indexing, means is already for this param
                yerr=uncertainties,  # No need for indexing, uncertainties is already for this param
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
                param_name,
                fontsize=self.plot_config.label_font_size,
            )
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            ax.grid(True, alpha=0.3)

        fig.suptitle('MOT Cloud Parameters', fontsize=self.plot_config.title_font_size)

        figname = f"{self.folder_path}\mot_params.png"
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
