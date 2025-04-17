from typing import Literal, Optional
from typing_extensions import assert_never

import h5py
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
import uncertainties.unumpy as unumpy
from matplotlib.figure import Figure
from pathlib import Path
from scipy import optimize
from scipy.constants import k as k_B, pi

try:
    lyse
except NameError:
    import lyse # needed for MLOOP

from .constants import cesium_atomic_mass
from .plot_config import PlotConfig
from .base_statistics import BaseStatistician

globals_friendly_names = {
    'bm_tof_imaging_delay': 'Time of flight',
    'mw_detuning': 'Microwave detuning',
}


class BulkGasStatistician(BaseStatistician):
    """Class for statistical analysis of bulk gas imaging data.

    This class provides methods for statistical analysis of bulk gas imaging data.
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
        super().__init__()
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(preproc_h5_path)
        self._save_mloop_params(shot_h5_path)
        self.folder_path = Path(preproc_h5_path).parent

    def _load_processed_quantities(self, preproc_h5_path):
        """Load processed quantities from an h5 file.

        Parameters
        ----------
        preproc_h5_path : str
            Path to the processed quantities h5 file
        """
        with h5py.File(preproc_h5_path, 'r') as f:
            self.atom_numbers = f['atom_numbers'][:, 0]
            self.params_list = f['params'][:]
            self.n_runs = f.attrs['n_runs']
            self.current_params = f['current_params'][:]
            if 'gaussian_cloud_params_nom' in f:
                # presently, only extract the params from the first fit
                self.gaussian_cloud_params_nom = f['gaussian_cloud_params_nom'][:, 0]
                self.gaussian_cloud_params_cov = f['gaussian_cloud_params_cov'][:, 0]

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

    @property
    def shots_processed(self) -> int:
        return self.atom_numbers.shape[0]

    @property
    def is_final_shot(self) -> bool:
        return self.shots_processed == self.n_runs

    def get_mean_std_of_unique_params(self, data, loop_params, unique_params):
        """
        Returns the sum of the data for each unique parameter combination.

        Parameters
        ----------
        data : array_like
            The data to sum.
        loop_params : array_like
            The parameters to loop over.
        unique_params : array_like
            The unique parameter combinations.

        Returns
        -------
        data_sum : array_like
            The sum of the data for each unique parameter combination.
        """

        means = [np.mean(data[np.where((loop_params == tuple(x)).all(axis=1))[0]])
            for x in unique_params
        ]

        stds = [np.std(data[np.where((loop_params == tuple(x)).all(axis=1))[0]])
            for x in unique_params
        ]

        means = np.array(means)
        stds = np.array(stds)

        return means, stds

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

        if self.current_params.shape[1] == 1:
            loop_params = self.current_params[:, 0]
        else:
            loop_params = self.current_params
        unique_params = np.unique(loop_params, axis = 0)

        if loop_params.size == 0:
            print("loop_params is empty with dimension", loop_params.ndim)
            print("atom number is", self.atom_numbers)
            if fig is not None:
                ax = fig.subplots()
                is_subfig = True

            ax.plot(self.atom_numbers, 'o')

            ax.set_title(
                self.folder_path,
                fontsize=8,
            )

            # ax.set_xlabel(
            #     f"{self.params_list[0][0].decode('utf-8')} [{self.params_list[0][1].decode('utf-8')}]",
            #     fontsize=self.plot_config.label_font_size,
            # )
            ax.set_ylabel(
                'atom count',
                fontsize=self.plot_config.label_font_size,
            )

            # ax.set_ylim(bottom=0)

            ax.grid(color=self.plot_config.grid_color_major, which='major')
            ax.grid(color=self.plot_config.grid_color_minor, which='minor')
        elif loop_params.ndim == 1:
            if fig is not None:
                ax = fig.subplots()
                is_subfig = True
            means = np.array([
                np.mean(self.atom_numbers[loop_params == x])
                for x in unique_params
            ])
            stds = np.array([
                np.std(self.atom_numbers[loop_params == x])
                for x in unique_params
            ])
            ns = np.array([
                np.sum(loop_params == x)
                for x in unique_params
            ])

            ax.set_title(
                self.folder_path,
                fontsize=8,
            )

            ax.errorbar(
                unique_params,
                means,
                yerr=stds/np.sqrt(ns),
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
            if self.is_final_shot and plot_lorentz:
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
        elif loop_params.ndim == 2:
            if fig is not None:
                ax1, ax2 = fig.subplots(2, 1)
                is_subfig = True

            x_params_index, y_params_index = self.get_params_order(unique_params)

            x_params = self.get_unique_params_along_axis(unique_params, x_params_index)
            y_params = self.get_unique_params_along_axis(unique_params, y_params_index)


            means, stds = self.get_mean_std_of_unique_params(
                self.atom_numbers,
                loop_params,
                unique_params,
            )

            means = self.reshape_to_unique_params_dim(means, x_params, y_params)
            stds = self.reshape_to_unique_params_dim(stds, x_params, y_params)

            x_params, y_params = np.meshgrid(x_params, y_params)

            pcolor_survival_rate = ax1.pcolormesh(
                x_params,
                y_params,
                means,
            )

            fig.colorbar(pcolor_survival_rate, ax=ax1)

            pcolor_std =ax2.pcolormesh(
                x_params,
                y_params,
                stds,
            )

            fig.colorbar(pcolor_std, ax=ax2)

            for ax in [ax1, ax2]:
                ax.set_xlabel(
                    f"{self.params_list[x_params_index][0].decode('utf-8')} [{self.params_list[x_params_index][1].decode('utf-8')}]",
                    fontsize=self.plot_config.label_font_size,
                )
                ax.set_ylabel(
                    f"{self.params_list[y_params_index][0].decode('utf-8')} [{self.params_list[y_params_index][1].decode('utf-8')}]",
                    fontsize=self.plot_config.label_font_size,
                )
                ax.tick_params(
                    axis='both',
                    which='major',
                    labelsize=self.plot_config.label_font_size,
                )
                ax.grid(color=self.plot_config.grid_color_major, which='major')
                # ax.grid(color=self.plot_config.grid_color_minor, which='minor')
            ax1.set_title('Mean', fontsize=self.plot_config.title_font_size)
            # ax1.set_yscale('log')
            ax2.set_title('Std', fontsize=self.plot_config.title_font_size)
        else:
            raise NotImplementedError("I only know how to plot 1d and 2d scans")

        if not is_subfig:
            fig.savefig(self.folder_path / 'count_vs_param.pdf')

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
        is_subfig = (fig is not None)

        axs = fig.subplots(2, 2, sharex=True)
        axs_flat = axs.flatten()

        ax_inds = [0, 0, 1, 1, 2, 3, 3]
        colors = ['C0', 'C1', 'C0', 'C1', 'C0', 'C0', 'C1']
        plot_labels = ['$x$', '$y$', '$\sigma_u$', '$\sigma_v$', None, 'peak height', 'offset']
        scale_factors = [1e3, 1e3, pi/2, 1]

        # Convert nominal values and uncertainties to ufloat arrays
        # shape: (n_shots, n_params = 6)
        gaussian_cloud_params = np.array([
            uncertainties.correlated_values(nom, cov)
            for nom, cov in zip(self.gaussian_cloud_params_nom, self.gaussian_cloud_params_cov)
        ])

        # For each parameter
        # for param_idx, (ax, param_name) in enumerate(zip(axs, param_names)):
        for param_idx in range(gaussian_cloud_params.shape[-1]):
            # For each unique x value, compute mean and combined uncertainty
            means = []
            uncerts = []
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
                uncerts.append(total_uncertainty)

            means = np.array(means)
            uncerts = np.array(uncerts)

            ax = axs_flat[ax_inds[param_idx]]
            scale_factor = scale_factors[ax_inds[param_idx]]

            # Plot with error bars
            ax.errorbar(
                unique_params,
                scale_factor * means,  # No need for indexing, means is already for this param
                yerr=scale_factor * uncerts,  # No need for indexing, uncertainties is already for this param
                color=colors[param_idx],
                marker='.',
                linestyle=('None' if param_idx in [2, 3] else 'solid'),
                alpha=0.5,
                capsize=3,
                label=plot_labels[param_idx],
            )

        loop_global_name: str = self.params_list[0][0].decode('utf-8')
        loop_global_unit: str = self.params_list[0][1].decode('utf-8')
        fig.supxlabel(
            f'{loop_global_name} [{loop_global_unit}]',
            fontsize=self.plot_config.label_font_size,
        )
        fig.suptitle('MOT Cloud Parameters', fontsize=self.plot_config.title_font_size)

        # time of flight temperature measurement
        if self.is_final_shot and loop_global_name == 'bm_tof_imaging_delay':
            times = self.current_params[:, 0]

            for param_idx in (2, 3):
                # x, y widths in fields (2, 3)
                widths_u = gaussian_cloud_params[:, param_idx]
                upopt = self.time_of_flight_fit(times, widths_u, method='curve_fit')
                init_width_u, temperature_u = upopt

                time_range = np.linspace(np.min(times), np.max(times), 101)
                axs_flat[1].plot(
                    time_range,
                    self.time_of_flight_expansion(time_range, *unumpy.nominal_values(upopt)) * scale_factors[ax_inds[param_idx]],
                    alpha=0.5,
                    color=colors[param_idx],
                    label=f'{1e+6 * temperature_u:S} $\mu$K'
                )

        for ax in axs_flat:
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

        axs[0, 0].set_ylabel('Position (mm)')
        axs[0, 1].set_ylabel('Gaussian width (mm)')
        axs[1, 0].set_ylabel('Rotation (deg)')
        axs[1, 1].set_ylabel('Counts')

        figname = f"{self.folder_path}\mot_params.png"
        if is_subfig:
            self.save_subfig(fig, figname)
        else:
            fig.savefig(figname)

    @staticmethod
    def time_of_flight_expansion(time, initial_width, temperature):
        return np.sqrt(initial_width**2 + (k_B * temperature / cesium_atomic_mass) * time**2)

    @classmethod
    def time_of_flight_fit(cls, times, widths_u, method: Literal['curve_fit', 'polyfit'] = 'polyfit'):
        '''
        See e.g. Tomasz M Brzozowski et al 2002 J. Opt. B: Quantum Semiclass. Opt. 4 62

        Parameters
        ----------
        times : np.ndarray
            Array of times
        widths_u : array_like of UFloats
            Array of widths
        method : {'curve_fit', 'polyfit'}
            Method of fitting
        '''
        if method == 'curve_fit':
            popt, pcov = optimize.curve_fit(
                cls.time_of_flight_expansion,
                times,
                unumpy.nominal_values(widths_u),
                sigma=unumpy.std_devs(widths_u),
                p0=(0.5 * widths_u[0].n, 25e-6),
            )
            init_width_u, temperature_u = uncertainties.correlated_values(popt, pcov)
            return init_width_u, temperature_u
        elif method == 'polyfit':
            square_widths_u = widths_u**2
            popt, pcov = np.polyfit(
                times ** 2,
                unumpy.nominal_values(square_widths_u),
                deg=1,
                w=1/unumpy.std_devs(square_widths_u),
                full=False,
                cov=True,
            )
            slope_u, initial_width_sq_u = uncertainties.correlated_values(popt, pcov)
            return unumpy.sqrt(initial_width_sq_u).item(), slope_u * cesium_atomic_mass / k_B
        else:
            assert_never(method)
