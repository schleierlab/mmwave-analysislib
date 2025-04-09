from typing import Literal, Optional
from typing_extensions import assert_never

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import beta, norm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from pathlib import Path
import lyse

# try:
#     lyse
# except NameError:
#     import lyse # needed for MLOOP

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

    site_occupancies: np.ndarray
    '''
    site_occupancies is of shape (num_shots, num_images, num_sites)
    '''

    def __init__(self,
                 preproc_h5_path: str,
                 shot_h5_path: Optional[str] = None,
                 plot_config: PlotConfig = None,
                 ):
        super().__init__()
        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(preproc_h5_path)
        if shot_h5_path is not None:
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

    @staticmethod
    def survival_fraction_bayesian(
            n_survived,
            n_total,
            center_method: Literal['mean', 'median', 'mode'],
            prior: Literal['uniform', 'jeffreys', 'haldane'],
            interval_method: Literal['centered', 'variance'],
            interval_sigmas: float = 1,
    ) -> tuple[float, tuple[float, float]]:
        '''
        Estimate a survival rate and errorbars for same
        by Bayesian inference, assuming a beta distribution prior
        (the conjugate prior for the Bernoulli distribution);
        the prior is specified by the caller.
        The survival rate is a measure of central tendency on the posterior distribution
        specified by the caller; the errorbars subtend the credible interval
        of user-specified width.

        Parameters
        ----------
        n_survived, n_total: int
            Number of survived and total atoms, respectively.
        center_method: {'mean', 'median'}
            Method for determining the center of the posterior distribution.
        prior: {'uniform', 'jeffreys', 'haldane'}
            Prior distribution for the beta distribution.
                * 'uniform' gives a uniform prior, Beta(1, 1).
                * 'jeffreys' gives the Jeffreys prior, which is just the arcsine distribution Beta(0.5, 0.5)
                * 'haldane' gives the (improper) Haldane prior, Beta(0, 0).
            See https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference
        interval_method: {'centered', 'variance'}
            Method for determining the credible interval.
        interval_sigmas: float
            Probability mass enclosed in the credible interval,
            parametrized by the number of standard deviations
            from the center of a Gaussian distribution that would enclose the same probability mass.
            Thus, interval_sigmas=1 corresponds to a ~68% credible interval.

        Returns
        -------
        center: float
            Center of the posterior distribution, calculated per `center_method`.
        yerr: tuple[float, float]
            Distance of the credible interval from the center.

        Notes
        -----
        mean, haldane recovers the naive estimate successes/total
        mean, uniform recovers Laplace's rule of succession.

        '''
        prior_shape_param: float
        if prior == 'uniform':
            prior_shape_param = 1
        elif prior == 'jeffreys':
            prior_shape_param = 0.5
        elif prior == 'haldane':
            prior_shape_param = 0

        posterior_beta = beta(
            n_survived + prior_shape_param,
            n_total - n_survived + prior_shape_param,
        )

        if center_method == 'mean':
            center = posterior_beta.mean()
        elif center_method == 'median':
            center = posterior_beta.median()
        else:
            assert_never(center_method)

        if interval_method == 'centered':
            tail_prob = norm.cdf(-interval_sigmas)
            credible_interval = posterior_beta.ppf([tail_prob, 1 - tail_prob])
            yerr = (credible_interval - center) * [-1, 1]
        elif interval_method == 'variance':
            yerr = posterior_beta.std() * interval_sigmas
        else:
            assert_never(interval_method)

        return center, yerr

    # @staticmethod
    # def survival_fraction(
    #     n_survived,
    #     n_total,
    #     method: Literal['exact', 'laplace'],
    #     uncertainty_method: Literal['wald', 'jeffreys', 'clopperpearson'],
    # ):
    #     if method == 'exact':
    #         survival_rate = n_survived / n_total
    #     elif method == 'laplace':
    #         survival_rate = (n_survived + 1) / (n_total + 2)
    #     elif method == 'median':
    #         survival_rate = beta(0.5, 0.5).median()
    #     else:
    #         assert_never(method)

    #     if uncertainty_method == 'wald':
    #         uncertainty = np.sqrt(survival_rate * (1 - survival_rate) / n_total)
    #         return survival_rate, uncertainty
    #     elif uncertainty_method == 'jeffreys':
    #         uncertainty = np.sqrt((n_survived + 1) / (n_total + 2)) / (n_total + 2)
    #     elif uncertainty_method == 'clopperpearson':
    #         tail_probability = norm.cdf(-1)  # 1-sigma 1-sided tail probability ~ (100% - 68%) / 2
    #         lolim = beta.ppf(tail_probability, n_survived, n_total - n_survived + 1)
    #         uplim = beta.ppf(1 - tail_probability, n_survived + 1, n_total - n_survived)
    #         return survival_rate, (survival_rate - lolim, uplim - survival_rate)
    #     else:
    #         assert_never(uncertainty_method)

    #     return survival_rate, uncertainty

    # @staticmethod
    # def survival_fraction_uncertainty(n_survived, n_total, method: Literal['frequentist', 'beta']):
    #     '''
    #     https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    #     '''
    #     if method == 'frequentist':
    #         p = n_survived / n_total
    #         return np.sqrt(p * (1 - p) / n_total)
    #     elif method == 'laplace':
    #         return np.sqrt((n_survived + 1) / (n_total + 2)) / (n_total + 2)
    #     else:
    #         assert_never(method)

    def get_sum_of_unique_params(self, data, loop_params, unique_params):
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

        data_sum = np.array([
            np.sum(data[np.where((loop_params == tuple(x)).all(axis=1))[0]])
            for x in unique_params
        ])

        return data_sum

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
            The index of the first parameter.
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

        data_new = np.pad(
            data,
            (0, x_params.shape[0]*y_params.shape[0] - data.shape[0]),
            'constant',
            constant_values=np.nan,
            ) # pad with nan if not enough data

        data_new = data_new.reshape([
            y_params.shape[0],
            x_params.shape[0]]
            ) # reshape

        return data_new





    def plot_survival_rate(self, fig: Optional[Figure] = None, plot_lorentz: bool = True):
        """
        Plots the total survival rate of atoms in the tweezers, summed over all sites.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        # Group data points by x-value and calculate statistics
        loop_params = self.current_params
        unique_params = np.unique(loop_params, axis = 0)
        # Calculate survival rates
        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=-1)

        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.product(self.site_occupancies[:, :2, :], axis=1).sum(axis=-1)

        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
            is_subfig = False

        if loop_params.ndim == 1:
            if fig is not None:
                ax = fig.subplots()
                is_subfig = True

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

            survival_rates, sigma_beta = self.survival_fraction_bayesian(
                surviving_atoms_sum,
                initial_atoms_sum,
                center_method='mean',
                prior='uniform',
                interval_method='variance',
            )

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

        elif loop_params.ndim == 2:
            if fig is not None:
                ax1, ax2 = fig.subplots(2, 1)
                is_subfig = True

            initial_atoms_sum =self.get_sum_of_unique_params(initial_atoms, loop_params, unique_params)
            surviving_atoms_sum = self.get_sum_of_unique_params(surviving_atoms, loop_params, unique_params)


            survival_rates = surviving_atoms_sum / initial_atoms_sum # simple survival rate
            # sqrt of variance of the posterior beta distribution
            sigma_beta = np.sqrt(survival_rates * (1 - survival_rates) )/ initial_atoms_sum # simple survival rate std

            x_params_index, y_params_index = self.get_params_order(unique_params)

            x_params = self.get_unique_params_along_axis(unique_params, x_params_index)
            y_params = self.get_unique_params_along_axis(unique_params, y_params_index)

            survival_rates = self.reshape_to_unique_params_dim(survival_rates, x_params, y_params)
            sigma_beta = self.reshape_to_unique_params_dim(sigma_beta, x_params, y_params)

            x_params, y_params = np.meshgrid(x_params, y_params)

            pcolor_survival_rate = ax1.pcolormesh(
                x_params,
                y_params,
                survival_rates,
            )

            fig.colorbar(pcolor_survival_rate, ax=ax1)

            pcolor_std =ax2.pcolormesh(
                x_params,
                y_params,
                sigma_beta,
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
                ax.grid(color=self.plot_config.grid_color_minor, which='minor')
            ax1.set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)
            ax2.set_title('Std over all sites', fontsize=self.plot_config.title_font_size)

        else:
            raise NotImplementedError("I only know how to plot 1d and 2d scans")


        figname = self.folder_path / 'survival_rate_vs_param.pdf'
        if not is_subfig:
            fig.savefig(figname)

    # TODO: this method needs updates that have already been applied to plot_survival_rate
    # Can redundant code here be consolidated with plot_survival_rate?
    def plot_survival_rate_by_site(self, ax: Optional[Axes] = None):
        """
        Plots the survival rate of atoms in the tweezers, site by site.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
        else:
            ax = ax

        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=0)
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.prod(self.site_occupancies[:, :2, :], axis=1).sum(axis=0)

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
        mean_survival_rate = np.sum(surviving_atoms)/np.sum(initial_atoms)
        ax.axhline(mean_survival_rate, color='red', linestyle='dashed', label=f'total = {mean_survival_rate*100:.1f}% ')
        ax.legend()

    # TODO: merge this into plot_survival_rate_by_site
    def plot_survival_rate_by_site_2d(self, fig: Optional[Figure] = None):
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

        loop_params = self.current_params[:, 0]

        # Group data points by x-value and calculate statistics
        unique_params = np.unique(loop_params)

        # Calculate survival rates
        initial_atoms = self.site_occupancies[:, 0, :]
        n_sites = self.site_occupancies.shape[2]

        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.product(self.site_occupancies[:, :2, :], axis=1)


        initial_atoms_sum = np.array([
            np.sum(initial_atoms[loop_params == x], axis=0)
            for x in unique_params
        ])

        surviving_atoms_sum = np.array([
            np.sum(surviving_atoms[loop_params == x], axis=0)
            for x in unique_params
        ])

        #survival_rates = surviving_atoms / initial_atoms
        # survival rate using laplace rule of succession
        survival_rates = (surviving_atoms_sum + 1) / (initial_atoms_sum + 2)

        ax.pcolormesh(
            unique_params,
            np.arange(n_sites),
            survival_rates.T,
        )

        fig.savefig(f"{self.folder_path}/survival_rate_by_site_2d.pdf")
