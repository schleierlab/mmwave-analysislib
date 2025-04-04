from typing import Optional
from matplotlib.figure import Figure
import h5py
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
import uncertainties.unumpy as unumpy
from pathlib import Path

try:
    lyse
except NameError:
    import lyse # needed for MLOOP

from .plot_config import PlotConfig
from .image import ROI
from .base_statistics import BaseStatistician


class TweezerStatistician(BaseStatistician):
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
                 plot_config: PlotConfig = None,
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
            self.camera_counts = f['camera_counts'][:]
            self.site_occupancies = f['site_occupancies'][:]
            self.site_rois = ROI.fromarray(f['site_rois'])
            self.params_list = f['params'][:]
            self.n_runs = f.attrs['n_runs']
            self.current_params = f['current_params'][:]

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
        survival_rate = 0; survival_uncertainty = 0.1
        mloop_result = (survival_rate, survival_uncertainty)
        run.save_results_dict(
            {
                'survival_rate': mloop_result if my_condition else (np.nan, np.nan),
            },
            uncertainties=True,
        )

    @property
    def shots_processed(self) -> int:
        return self.site_occupancies.shape[0]

    @property
    def is_final_shot(self) -> bool:
        return self.shots_processed == self.n_runs

    def plot_survival_rate(self, fig: Optional[Figure] = None, plot_lorentz: bool = True):
        """
        Plots the total survival rate of atoms in the tweezers, summed over all sites.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
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
        
        # Calculate survival rates
        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=-1)
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.product(self.site_occupancies[:, :2, :], axis=1).sum(axis=-1)

        initial_atoms_sum = np.array([
            np.sum(initial_atoms[loop_params == x])
            for x in unique_params
        ])

        surviving_atoms_sum = np.array([
            np.sum(surviving_atoms[loop_params == x])
            for x in unique_params
        ])

        #survival_rates = surviving_atoms / initial_atoms
        # survival rate using laplace rule of succession
        survival_rates = (surviving_atoms_sum + 1) / (initial_atoms_sum + 2)
        # sqrt of variance of the posterior beta distribution
        sigma_beta = np.sqrt((surviving_atoms_sum + 1) * (initial_atoms_sum - surviving_atoms_sum + 1) / ((initial_atoms_sum + 3) * (initial_atoms_sum + 2) ** 2))

        # Calculate means and stds for the unique parameters
        # means = np.array([
        #     np.mean(survival_rates[loop_params == x])
        #     for x in unique_params
        # ])
        # stds = np.array([
        #     np.std(survival_rates[loop_params == x])
        #     for x in unique_params
        # ])

        ax.errorbar(
            unique_params,
            survival_rates,
            yerr=sigma_beta,
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
            'Survival rate',
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
        ax.set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)

        # doing the fit at the end of the run
        if self.is_final_shot and plot_lorentz:
            popt, pcov = self.fit_lorentzian(unique_params, survival_rates, sigma=sigma_beta)
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

        figname = f"{self.folder_path}\survival_rate vs param.png"
        if is_subfig:
            self.save_subfig(fig, figname)
        else:
            fig.savefig(figname)

    # TODO: this method needs updates that have already been applied to plot_survival_rate
    # Can redundant code here be consolidated with plot_survival_rate?
    def plot_survival_rate_by_site(self, fig: Optional[Figure] = None):
        """
        Plots the survival rate of atoms in the tweezers, site by site.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
        else:
            ax = fig.subplots()

        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=0)
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.product(self.site_occupancies[:, :2, :], axis=1).sum(axis=0)

        survival_rates = surviving_atoms / initial_atoms
        ax.plot(
            np.arange(len(initial_atoms)),
            survival_rates,
            marker='.',
        )
        ax.set_xlabel('Site number', fontsize=self.plot_config.label_font_size)
        ax.set_ylabel('Survival rate', fontsize=self.plot_config.label_font_size)
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.plot_config.label_font_size,
        )
        ax.grid(color=self.plot_config.grid_color_major, which='major')
        ax.grid(color=self.plot_config.grid_color_minor, which='minor')
        ax.set_title('Survival rate by site', fontsize=self.plot_config.title_font_size)
