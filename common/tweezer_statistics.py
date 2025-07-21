from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional, overload
from typing_extensions import assert_never

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.typing as pdt
import uncertainties
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.stats import beta, norm
from scipy.optimize import curve_fit

from .plot_config import PlotConfig
from .image import ROI
from .base_statistics import BaseStatistician


@dataclass
class ScanningParameter:
    name: str
    unit: str
    friendly_name: Optional[str] = None

    def __str__(self) -> str:
        return self.name
    
    @classmethod
    def from_h5_tuple(cls, tup) -> ScanningParameter:
        name: bytes
        units: bytes
        expr: bytes
        name, units, expr = tup
        return cls(name.decode('utf-8'), units.decode('utf-8'))

    @property
    def axis_label(self):
        namestr = self.friendly_name if self.friendly_name is not None else self.name
        unitstr = f' ({self.unit})' if self.unit != '' else ''
        return f'{namestr}{unitstr}'


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

    # TODO clean this up
    KEY_SITE: ClassVar[str] = 'site'
    KEY_INITIAL: ClassVar[str] = 'initial'
    KEY_SURVIVAL: ClassVar[str] = 'survival'
    KEY_SURVIVAL_RATE: ClassVar[str] = 'survival_rate'
    KEY_SURVIVAL_RATE_STD: ClassVar[str] = 'survival_rate_std'

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

            self.params = [ScanningParameter.from_h5_tuple(tup) for tup in self.params_list]

    @property
    def initial_atoms_array(self):
        return self.site_occupancies[:, 0, :]
    
    @property
    def surviving_atoms_array(self):
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        return self.site_occupancies[:, :2, :].prod(axis=-2)

    def dataframe(self) -> pd.DataFrame:
        '''
        Return dataframe of the form:

                  mw_detuning  ryd_456_mirror_2_h  site  initial  survival
            0             2.6                 3.0     0      1.0       1.0
            1             2.6                 3.0     1      1.0       1.0
            2             2.6                 3.0     2      0.0       0.0
            3             2.6                 3.0     3      0.0       0.0
            4             2.6                 3.0     4      1.0       0.0
            ...           ...                 ...   ...      ...       ...

        The columns are: [*scanned_globals, site index, initial, survival].
        There are n_sites rows per shot, for a total of n_sites * n_shots rows.
        This form is amenable to grouping (via `.groupby()`) and aggregation.
        '''
        index = pd.MultiIndex.from_arrays(
            self.current_params.T,
            names=[param.name for param in self.params],
        )

        def assemble_occupancy_df(array: NDArray, name: str):
            df = pd.DataFrame(array, index=index)
            df.columns.name = self.KEY_SITE
            occupancy_df = df.stack()
            occupancy_df.name = name
            return occupancy_df
        
        df_initial = assemble_occupancy_df(self.site_occupancies[:, 0, :], name=self.KEY_INITIAL)
        df_survival = assemble_occupancy_df(self.site_occupancies[..., :2, :].prod(axis=-2), name=self.KEY_SURVIVAL)
        df = pd.concat([df_initial, df_survival], axis=1)

        return df.reset_index()
    
    @overload
    def dataframe_survival(self, data: pd.DataFrame) -> pd.Series: ...
    @overload
    def dataframe_survival(self, data: pdt.DataFrameGroupBy) -> pd.DataFrame: ...
    
    def dataframe_survival(self, data):
        df = data[[self.KEY_INITIAL, self.KEY_SURVIVAL]].sum()

        surv = df[self.KEY_SURVIVAL]
        total = df[self.KEY_INITIAL]
        df[self.KEY_SURVIVAL_RATE] = surv / total

        laplace_p = (surv + 1) / (total + 2)
        df[self.KEY_SURVIVAL_RATE_STD] = np.sqrt(laplace_p * (1 - laplace_p) / (total + 2))

        return df

    def rearrange_success_rate(self, target_array):
        atom_number_target_array = np.zeros(len(self.site_occupancies[:,0,0]))
        rearrange_index = []

        for i in np.arange(atom_number_target_array.shape[0]):
            first_shot = self.site_occupancies[i,0,:]
            second_shot = self.site_occupancies[i,1,:]
            second_shot_target_array = second_shot[target_array]
            # use only the targe indexs
            if np.sum(first_shot) >= len(target_array):
                # print("Occupy >= 50%, starting rearrange")
                rearrange_index.append(i)
                atom_number_target_array[i] = np.sum(second_shot_target_array)
                if atom_number_target_array[i] == 0:
                    print("Warning, the rearrangement happens but 0 atom left in the target array, something is wrong!")

        # second_shot_target_array_lst = self.site_occupancies[rearrange_index,1,target_array]
        # print(second_shot_target_array_lst.shape)

        return atom_number_target_array

    def rearragne_statistics(self, target_array):
        n_target = len(target_array)
        # Sum over atoms for each shot, for the first image (axis=1 is atoms)
        first_img_atom_counts = self.site_occupancies[:, 0, :].sum(axis=1)  # shape: (num_shots,)
        rearrange_shots = np.where(first_img_atom_counts >= n_target)[0].tolist()
        n_rearrange_shots = len(rearrange_shots)

        # Create zero array of the same shape
        target_array_boolean = np.zeros_like(self.site_occupancies[rearrange_shots, 0, :])

        # Set target sites to 1 for all selected shots
        target_array_boolean[:, target_array] = 1

        site_success_rate = self.site_occupancies[rearrange_shots, 1, :]/target_array_boolean # shape: (num_shots, num_sites)
        avg_site_success_rate = np.mean(site_success_rate[:, target_array], axis=0) # shape: (num_sites,)

        # For each shot in rearrange_shots, sum over all target sites in the 2nd image
        # Shape: (n_rearrange_shots, len(target_array)) -> sum over axis=1 -> (n_rearrange_shots,)
        # rearrange_success_atom_count = self.site_occupancies[rearrange_shots, 1, :][:, target_array].sum(axis=1)

        # Calculate atom count in target_array for the second image, for ALL shots
        atom_count_in_target_all_shots = self.site_occupancies[:, 1, :][:, target_array].sum(axis=1)
        # Filter this count for only the rearrange_shots
        atom_count_in_target_rearrange_shots = atom_count_in_target_all_shots[rearrange_shots] # shape: (n_rearrange_shots,)

        success_rearrange = np.sum(atom_count_in_target_rearrange_shots == n_target)
        atom_count_in_target = [atom_count_in_target_all_shots, atom_count_in_target_rearrange_shots]
        return success_rearrange, atom_count_in_target, n_rearrange_shots, avg_site_success_rate

    def plot_rearrange_histagram(self, target_array, ax: Optional[Axes] = None, plot_overlapping_histograms: bool = True):
        '''
        Plots a histogram of the number of sites in the taerget array after rearrangement.

        Parameters
        ----------
        target_array : array_like
            Array of target sites.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure is created.
        plot_overlapping_histograms : bool, optional
            Whether to plot overlapping histograms. The default is True.
            When set to True, plot both the histogram of all shots and the histogram of rearrange shots.
            This is helpful when we have bug that causes a lot of rearrangement shots end up with zero atoms in target sites.
            When set to False, plot only the histogram of all shots.
        '''
        # Bar plot: Number of sites after rearrangement
        success_rearrange, atom_count_in_target_list, n_rearrange_shots, _ = self.rearragne_statistics(target_array)

        atom_counts_all_shots = atom_count_in_target_list[0]
        n_target = len(target_array)
        n_shots = len(self.site_occupancies) # For unified title

        ax.set_xlabel('Number of loaded target sites after rearrangement')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y')

        title_parts = [f'{n_target} target sites']
        if n_shots > 0 and n_rearrange_shots > 0:
            ratio = n_rearrange_shots / n_shots
            rate = success_rearrange / n_rearrange_shots
            title_parts.append(f'rearrange shot ratio: {ratio:.3f}, success rate: {rate:.3f}')
        elif n_rearrange_shots == 0 and n_shots > 0:
             title_parts.append('no rearrangement attempts made')
        else:
            title_parts.append('no shot data for title metrics')
        ax.set_title('\n'.join(title_parts))

        if plot_overlapping_histograms:
            atom_counts_rearrange_shots = atom_count_in_target_list[1]

            unique_all, counts_all = np.unique(atom_counts_all_shots, return_counts=True)
            unique_rearrange, counts_rearrange = np.unique(atom_counts_rearrange_shots, return_counts=True)

            bar_width_all = 0.8
            bar_width_rearrange = bar_width_all * 0.7

            x_all = unique_all.astype(int)
            x_rearrange = unique_rearrange.astype(int)

            if len(counts_all) > 0:
                ax.bar(x_all, counts_all, width=bar_width_all, label=f'All Shots ({n_shots} shots)', alpha=0.5, color='skyblue')
            if len(counts_rearrange) > 0:
                ax.bar(x_rearrange, counts_rearrange, width=bar_width_rearrange, label=f'Rearrange Attempts ({n_rearrange_shots} shots)', alpha=0.8, color='royalblue')

            if len(x_all) > 0:
                ax.set_xticks(x_all)

            if len(counts_all) > 0 or len(counts_rearrange) > 0:
                ax.legend()
        else:
            unique_elements, counts = np.unique(atom_counts_all_shots, return_counts=True)

            if len(counts) > 0:
                ax.bar(unique_elements, counts, width=0.5)

            if len(unique_elements) > 0:
                ax.set_xticks(unique_elements.astype(int))

        print('n_shots (total experiment)', n_shots)
        print('n_rarrange_shots', n_rearrange_shots)
        print('success_rearrange', success_rearrange)
        print(f'Rearrange attempts with 0 loaded atoms: {(atom_count_in_target_list[1] == 0).sum()}')

    def plot_rearrange_site_success_rate(self, target_array, ax: Optional[Axes] = None):
        # Site success rate plot
        _, _, n_rearrange_shots, avg_site_success_rate = self.rearragne_statistics(target_array)

        # n_sites = self.site_occupancies.shape[2]
        ax.plot(target_array, avg_site_success_rate, 'o')
        ax.axhline(np.mean(avg_site_success_rate), color='red', linestyle='dashed', label=f'mean = {np.mean(avg_site_success_rate):.3f}')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Tweezer index')
        ax.set_ylabel('Rearrangement success rate')
        ax.set_title(f'Target sites success rate, {n_rearrange_shots} shots average')
        # Make x-axis show only integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_site_loading_rates(self, ax: Optional[Axes] = None):
        first_img_atoms_by_site = self.site_occupancies[:, 0, :].sum(axis=0) # sum over all shots for the first image, shape: (num_sites,)
        second_img_atoms_by_site = self.site_occupancies[:, 1, :].sum(axis=0) # sum over all shots for the second image
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        n_shots = len(self.site_occupancies)
        first_img_loading_rate = first_img_atoms_by_site/n_shots
        second_img_loading_rate = second_img_atoms_by_site/n_shots
        ax.plot(first_img_loading_rate, '.-', label='1st shot')
        ax.plot(second_img_loading_rate,  '.-', label='2nd shot')
        ax.grid()
        ax.set_xlabel('Tweezer index')
        ax.set_ylabel('loading rate')
        ax.set_title(f'Tweezer site loading rates, {n_shots} shots average')
        ax.legend()

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
        import lyse
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

    def plot_target_sites_success_rate(self, target_array, fig: Optional[Figure] = None):
        """
        Plots the total survival rate of atoms in the tweezers, summed over all sites.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        # Calculate survival rates
        if fig is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
            is_subfig = False
        else:
            ax = fig.subplots()
            is_subfig = True

        atom_number_target_array = self.rearrange_success_rate(target_array)

        target_sites_success_rate =  atom_number_target_array/target_array.shape[0]

        error = np.sqrt(
            (target_sites_success_rate * (1 - target_sites_success_rate))
            / atom_number_target_array[atom_number_target_array!=0].shape[0]
            )

        ax.set_title(
            self.folder_path,
            fontsize=8,
        )

        ax.errorbar(
            x = np.arange(self.site_occupancies.shape[0]),
            y = target_sites_success_rate,
            yerr=error,
            marker='.',
            linestyle='-',
            alpha=0.5,
            capsize=3,
        )
        ax.set_ylabel(
            'Target sites success rate',
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
        ax.set_title('Target sites success rate over all sites', fontsize=self.plot_config.title_font_size)

        figname = self.folder_path / 'target_sites_success_rate.pdf'
        if not is_subfig:
            fig.savefig(figname)

    def plot_survival_rate_2d(
            self,
            fig: Figure,
            plot_gaussian: bool = False,
    ):
        if fig is not None:
            ax1, ax2 = fig.subplots(2, 1)

        loop_params = self._loop_params()
        unique_params = self.unique_params()

        x_params_index, y_params_index = self.get_params_order(unique_params)
        x_params = self.get_unique_params_along_axis(unique_params, x_params_index)
        y_params = self.get_unique_params_along_axis(unique_params, y_params_index)

        # Calculate survival rates
        initial_atoms = self.initial_atoms_array.sum(axis=-1) # sum over all sites for each shot
        surviving_atoms = self.surviving_atoms_array.sum(axis=-1)

        initial_atoms_sum = self.get_sum_of_unique_params(initial_atoms, loop_params, unique_params)
        surviving_atoms_sum = self.get_sum_of_unique_params(surviving_atoms, loop_params, unique_params)

        survival_rates = surviving_atoms_sum / initial_atoms_sum # simple survival rate
        sigma_beta = np.sqrt(survival_rates * (1 - survival_rates)) / initial_atoms_sum # simple survival rate std

        survival_rates = self.reshape_to_unique_params_dim(survival_rates, x_params, y_params)
        sigma_beta = self.reshape_to_unique_params_dim(sigma_beta, x_params, y_params)

        x_params, y_params = np.meshgrid(x_params, y_params)
        
        df = self.dataframe()
        groupby = df.groupby([param.name for param in self.params])
        survival_df = self.dataframe_survival(groupby)

        def plot_key_2d(df, ax):
            unstack = df.unstack()
            return ax.pcolormesh(
                unstack.columns,
                unstack.index,
                unstack,
            )

        pcolor_survival_rate = plot_key_2d(survival_df[self.KEY_SURVIVAL_RATE], ax1)
        pcolor_std = plot_key_2d(survival_df[self.KEY_SURVIVAL_RATE_STD], ax2)

        fig.colorbar(pcolor_survival_rate, ax=ax1)
        fig.colorbar(pcolor_std, ax=ax2)

        for axs in [ax1, ax2]:
            axs.set_xlabel(
                self.params[x_params_index].axis_label,
                fontsize=self.plot_config.label_font_size,
            )
            axs.set_ylabel(
                self.params[y_params_index].axis_label,
                fontsize=self.plot_config.label_font_size,
            )
            axs.tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            axs.grid(color=self.plot_config.grid_color_major, which='major')
            axs.grid(color=self.plot_config.grid_color_minor, which='minor')
        ax1.set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)
        ax2.set_title('Std over all sites', fontsize=self.plot_config.title_font_size)
        if plot_gaussian:
            popt, pcov = self.fit_gaussian_2d(x_params, y_params, survival_rates)
            perr = np.sqrt(np.diag(pcov))
            ax1.title.set_text(f'X waist = {popt[3]:.2f} +/- {perr[3]:.2f}, Y waist = {popt[4]:.2f} +/- {perr[4]:.2f}')

        return unique_params, survival_rates, sigma_beta

    def plot_survival_rate(self, fig: Optional[Figure] = None, plot_lorentz: bool = True, plot_gaussian: bool = False):
        """
        Plots the total survival rate of atoms in the tweezers, summed over all sites.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        loop_params = self._loop_params()
        unique_params = self.unique_params()

        # Calculate survival rates
        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=-1) # sum over all sites for each shot

        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.prod(self.site_occupancies[:, :2, :], axis=1).sum(axis=-1)

        is_subfig: bool = True
        if fig is None:
            fig, axs = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )
            is_subfig = False

        if loop_params.size == 0:
            print("loop_params is empty with dimension", loop_params.ndim)
            if fig is not None:
                axs = fig.subplots(nrows=2,ncols=1)

            survival_rates = surviving_atoms / initial_atoms
            loading_rates = initial_atoms/self.site_occupancies.shape[2]
            print("survival rate is", survival_rates)

            error = np.sqrt((survival_rates * (1 - survival_rates)) / self.site_occupancies.shape[2])
            loading_rates_error = np.sqrt((loading_rates * (1 - loading_rates)) / self.site_occupancies.shape[2])

            axs.set_title(
                self.folder_path,
                fontsize=8,
            )

            axs[0].errorbar(
                0,
                survival_rates,
                yerr=error,
                marker='.',
                linestyle='-',
                alpha=0.5,
                capsize=3,
            )
            axs[0].set_ylabel(
                'Survival rate',
                fontsize=self.plot_config.label_font_size,
            )
            axs[0].tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            axs[0].set_ylim(bottom=0)
            axs[0].grid(color=self.plot_config.grid_color_major, which='major')
            axs[0].grid(color=self.plot_config.grid_color_minor, which='minor')
            axs[0].set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)

            axs[1].errorbar(
                0,
                loading_rates,
                yerr = loading_rates_error,
                marker='.',
                linestyle='-',
                alpha=0.5,
                capsize=3,
                )
            axs[1].set_ylabel(
                'Loading rate',
                fontsize=self.plot_config.label_font_size,
            )
            axs[1].tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            axs[1].set_ylim(bottom=0)
            axs[1].grid(color=self.plot_config.grid_color_major, which='major')
            axs[1].grid(color=self.plot_config.grid_color_minor, which='minor')
            axs[1].set_title('Loading rate over all sites', fontsize=self.plot_config.title_font_size)

        elif loop_params.ndim == 1:
            if fig is not None:
                axs = fig.subplots(nrows=3, ncols=1)

            initial_atoms_sum = np.array([
                np.sum(initial_atoms[loop_params == x])
                for x in unique_params
            ])

            surviving_atoms_sum = np.array([
                np.sum(surviving_atoms[loop_params == x])
                for x in unique_params
            ])

            first_img_atom_counts = self.site_occupancies[:,0,:].sum(axis =1)
            rearrange_shots = first_img_atom_counts >= (self.site_occupancies[0,0,:].shape[0]/2)
            rearrange_shots_unique= np.array([
                np.sum(rearrange_shots[loop_params == x])
                for x in unique_params
                ])


            #survival_rates = surviving_atoms / initial_atoms
            # survival rate using laplace rule of succession
            survival_rates = (surviving_atoms_sum + 1) / (initial_atoms_sum + 2)
            # sqrt of variance of the posterior beta distribution
            sigma_beta = np.sqrt((surviving_atoms_sum + 1) * (initial_atoms_sum - surviving_atoms_sum + 1) / ((initial_atoms_sum + 3) * (initial_atoms_sum + 2) ** 2))


            n_rep = np.ceil(self.site_occupancies.shape[0]/unique_params.shape[0])
            n_sites = self.site_occupancies.shape[2]
            loading_rates = initial_atoms_sum/(n_rep*n_sites)
            loading_rates_error = np.sqrt((1-loading_rates)*loading_rates / (n_rep*n_sites))
            rearrange_rates = rearrange_shots_unique / n_rep
            rearrange_rates_error = np.sqrt((1-rearrange_rates)*rearrange_rates/n_rep)

            for ax in axs:
                ax.set_xlabel(
                    self.params[0].axis_label,
                    fontsize=self.plot_config.label_font_size,
                )
                ax.set_ylim(bottom=0)
                ax.grid(color=self.plot_config.grid_color_major, which='major')
                ax.grid(color=self.plot_config.grid_color_minor, which='minor')

            survival_rates, sigma_beta = self.survival_fraction_bayesian(
                surviving_atoms_sum,
                initial_atoms_sum,
                center_method='mean',
                prior='uniform',
                interval_method='variance',
            )

            fig.suptitle(
                self.folder_path,
                fontsize=8,
            )

            axs[0].errorbar(
                unique_params,
                survival_rates,
                yerr=sigma_beta,
                marker='.',
                linestyle='-',
                alpha=0.5,
                capsize=3,
            )

            axs[0].set_ylabel(
                'Survival rate',
                fontsize=self.plot_config.label_font_size,
            )
            axs[0].tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )

            axs[0].set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)

            axs[1].errorbar(
                unique_params,
                loading_rates,
                yerr=loading_rates_error,
                marker='.',
                linestyle='-',
                alpha=0.5,
                capsize=3,
            )

            axs[1].hlines(
                y = 0.5,
                xmin = np.min(unique_params),
                xmax = np.max(unique_params),
                linestyle = '--',
                color = 'r',
                label = '50%'
                )

            axs[1].set_ylabel(
                'Loading rate',
                fontsize=self.plot_config.label_font_size,
            )
            axs[1].set_title('Loading rate over all sites', fontsize=self.plot_config.title_font_size)
            axs[1].legend()


            axs[2].errorbar(
                unique_params,
                rearrange_rates,
                yerr=rearrange_rates_error,
                marker='.',
                linestyle='-',
                alpha=0.5,
                capsize=3,
            )

            axs[2].hlines(
                y = 0.5,
                xmin = np.min(unique_params),
                xmax = np.max(unique_params),
                linestyle = '--',
                color = 'r',
                label = '50%'
                )


            axs[2].set_ylabel(
                'Rearrange rate',
                fontsize=self.plot_config.label_font_size,
            )
            axs[2].set_title(f'Rearrange rate over all shots = {np.mean(rearrange_rates):.2}', fontsize=self.plot_config.title_font_size)
            axs[2].legend()

            # doing the fit at the end of the run
            if self.is_final_shot and plot_lorentz:
                popt, pcov = self.fit_lorentzian(unique_params, survival_rates, sigma=sigma_beta)
                upopt = uncertainties.correlated_values(popt, pcov)

                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1000,
                )

                axs[0].plot(x_plot, self.lorentzian(x_plot, *popt))
                fig.suptitle(
                    f'Center frequency: ${upopt[0]:SL}$ MHz; '
                    f'Width: ${1e+3 * upopt[1]:SL}$ kHz'
                )
            return unique_params, survival_rates, sigma_beta

        elif loop_params.ndim == 2:
            unique_params, survival_rates, sigma_beta = self.plot_survival_rate_2d(fig, plot_gaussian)
        else:
            raise NotImplementedError("I only know how to plot 1d and 2d scans")

        figname = self.folder_path / 'survival_rate_vs_param.pdf'
        if not is_subfig:
            fig.savefig(figname)
        return unique_params, survival_rates, sigma_beta

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

        initial_atoms = self.site_occupancies[:, 0, :].sum(axis=0) # sum over all shots for the first image
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

    def loop_param_and_site_survival_rate_matrix(self, num_time_groups = 1, method = 'laplace'):
        '''
        return an array of loop parameters
        and a array of matrix with each row being the survival rate array of each site
        with shape (num_sites, length_loop_params, num_time_groups)
        num_time_groups split the data into groups taken in earlier time and later time
        based on the shot number
        '''
        loop_params = self.current_params[:, 0]
        unique_params = np.unique(loop_params)
        num_unique = len(unique_params)
        num_sites = self.site_occupancies.shape[2]

        initial_atoms = self.site_occupancies[:, 0, :]
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = np.prod(self.site_occupancies[:, :2, :], axis=1)

        # Initialize array: (num_sites, num_unique_params, num_groups_per_param)
        survival_rates = np.empty((num_sites, num_unique, num_time_groups))

        for i, x in enumerate(unique_params):
            idx = np.where(loop_params == x)[0]
            shots_per_param = len(idx)
            group_size = shots_per_param // num_time_groups

            for g in range(num_time_groups):
                start = g * group_size
                end = (g + 1) * group_size if g < num_time_groups - 1 else shots_per_param

                selected_idx = idx[start:end]
                i_sum = np.sum(initial_atoms[selected_idx], axis=0)
                s_sum = np.sum(surviving_atoms[selected_idx], axis=0)
                # survival_rates = surviving_atoms / initial_atoms
                # survival rate using laplace rule of succession
                if method == 'exact':
                    survival_rates[:, i, g] = s_sum / i_sum
                elif method == 'laplace':
                    survival_rates[:, i, g] = (s_sum + 1) / (i_sum + 2)

        return unique_params, survival_rates

    # TODO: merge this into plot_survival_rate_by_site
    def plot_survival_rate_by_site_2d(self, ax: Optional[Figure] = None, plot_grouped_averaged = False): #TODO: add grouped averaged option
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
            is_subfig = False
        else:
            ax = ax
            is_subfig = True

        unique_params, survival_rates_matrix = self.loop_param_and_site_survival_rate_matrix()
        survival_rates_matrix = survival_rates_matrix[:, :, 0]

        n_sites = survival_rates_matrix.shape[0]

        if plot_grouped_averaged:
            n_groups, averaged_data = self.group_data(survival_rates_matrix, group_size = 10)
            # 2D plot, group averaged
            pm = ax.pcolormesh(
                unique_params,
                np.arange(n_groups),
                averaged_data,
            )
        else:
            # 2D plot, all sites
            pm = ax.pcolormesh(
                unique_params,
                np.arange(n_sites),
                survival_rates_matrix,
            )

        ax.set_xlabel(self.params[0].axis_label)
        ax.set_ylabel('Site index')
        cbar = fig.colorbar(pm, ax=ax)

        if not is_subfig:
            fig.savefig(f"{self.folder_path}/survival_rate_by_site_2d.pdf")
            fig.suptitle(f"{self.folder_path}")

    def group_data(self, data, group_size):
        n_groups = data.shape[0]//group_size
        print('n_groups',n_groups)
        grouped_data = data[:data.shape[0]].reshape(n_groups, group_size, -1)
        averaged_data = grouped_data.mean(axis = 1)

        print('shape of data', data.shape)
        print('shape of averaged data', averaged_data.shape)
        return n_groups, averaged_data

    def plot_avg_survival_rate_by_grouped_sites_1d(self, group_size, fit_type=None, num_time_groups = 1):
        """
        Parameters:
            unique_params: shape (num_unique_params,)
            data: shape (num_sites, num_unique_params, num_groups)
            group_size: how many sites per row (grouped for averaging)
            fit_type: if 'rabi_oscillation', fit each trace
        """
        unique_params, data = self.loop_param_and_site_survival_rate_matrix(num_time_groups)
        num_sites, num_params, num_groups = data.shape
        assert num_sites % group_size == 0, "num_sites must be divisible by group_size"
        n_rows = num_sites // group_size
        n_cols = num_groups

        # Average over site groups
        averaged_data = data.reshape(n_rows, group_size, num_params, num_groups).mean(axis=1)
        # shape: (n_rows, num_params, num_groups)

        # Create subplot grid
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True,
                                figsize=(4 * n_cols, 2.5 * n_rows), layout='constrained')

        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = np.expand_dims(axs, axis=0)  # shape (1, n_cols)
        elif n_cols == 1:
            axs = np.expand_dims(axs, axis=1)  # shape (n_rows, 1)
        # else axs is already 2D

        for row in range(n_rows):
            for col in range(n_cols):
                ax = axs[row, col]
                y = averaged_data[row, :, col]
                ax.plot(unique_params, y, '.-', label=rf'site group {row}')

                # Title for top row only
                if row == 0:
                    ax.set_title(f'Time Group {col}', fontsize=12)

                if fit_type == 'rabi_oscillation':
                    try:
                        initial_guess = [1, 2 * np.pi * 2e6, 0, 1e-6, 0.5]
                        params_opt, _ = curve_fit(self.rabi_model, unique_params, y, p0=initial_guess)
                        A_fit, Omega_fit, phi_fit, T2_fit, C_fit = params_opt
                        ax.plot(unique_params, self.rabi_model(unique_params, *params_opt), 'r-', label='Fit')

                        annotation_text = (
                            f'p-p Ampl: {A_fit*2:.3f}\n'
                            f'Ω: {Omega_fit / 1e6 / (2*np.pi):.3f} MHz\n'
                            f'Phase: {phi_fit:.2f} rad\n'
                            f'T₂*: {T2_fit * 1e6:.2f} µs'
                        )
                        ax.annotate(annotation_text,
                                    xy=(0.02, 0.05), xycoords='axes fraction',
                                    fontsize=9, ha='left', va='bottom')
                    except Exception as e:
                        ax.annotate("Fit failed", xy=(0.02, 0.05), xycoords='axes fraction',
                                    fontsize=9, ha='left', va='bottom')

                ax.legend(loc='upper right')

        fig.supxlabel('Time')
        fig.supylabel('Population')

    def plot_avg_survival_rate_by_grouped_sites_1d_old(self, group_size, fit_type = None):
        unique_params, data = self.loop_param_and_site_survival_rate_matrix()
        site_occupancies_matrix = self.site_occupancies
        file_path = os.path.join(f"{self.folder_path}/", 'survival_by_sites_matrix.npy')
        np.save(file_path, data)
        file_path = os.path.join(f"{self.folder_path}/", 'site_occupancies_matrix.npy')
        np.save(file_path, site_occupancies_matrix)
        print('files saved!')

        n_groups, averaged_data = self.group_data(data, group_size)

        # 1D plot, group averaged, in the same plot
        # for i in np.arange(averaged_data.shape[0]):
        #     ax.plot(unique_params, averaged_data[i],'.-',label = rf'{i}')

        #1D plot, group averaged, separate plots with fit
        fig, axs = plt.subplots(nrows=n_groups, ncols=1, sharex=True, sharey= True, layout='constrained')
        for i in np.flip(np.arange(averaged_data.shape[0])):
            ax = axs[-i-1]
            ax.plot(unique_params, averaged_data[i],'.-',label = rf'group {i} data')
            if fit_type == 'rabi_oscillation':
                # Fit the model to the data
                initial_guess = [1, 2*np.pi*1.6e6, 0, 3e-6, 0.5]
                params_opt, params_cov = curve_fit(self.rabi_model, unique_params, averaged_data[i], p0=initial_guess)

                # Extract fit results
                A_fit, Omega_fit, phi_fit, T2_fit, C_fit = params_opt

                upopt = uncertainties.correlated_values(params_opt, params_cov)


                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1000,
                )
                ax.plot(x_plot, self.rabi_model(x_plot, *params_opt), 'r-', label=rf'{i}Fit')
                annotation_text = (
                    f'p-p Ampl: {A_fit*2:.3f}\n'
                    f'$\Omega$: {upopt[1] / 1e6 / (2 * np.pi):S} MHz\n'
                    f'Phase: {phi_fit:.2f} rad\n'
                    f'$T_2^*$: {upopt[3] * 1e6 :S} µs\n'
                    # f'Offset: {C_fit:.2f}'
                )
                # ax.annotate(annotation_text,
                #             xy=(0.02, 0.95),  # top-left corner inside the subplot
                #             xycoords='axes fraction',
                #             fontsize=9,
                #             ha='left', va='top',
                #             )
                ax.annotate(
                    annotation_text,
                    xy=(0.02, 0.05),  # Changed to bottom-left corner (x=0.02, y=0.05)
                    xycoords='axes fraction',
                    fontsize=9,
                    ha='left',
                    va='bottom',
                )

                ax.legend(loc='upper right')
            elif fit_type == 'lorentzian':
                popt, pcov = self.fit_lorentzian(unique_params, averaged_data[i], sigma=None, peak_direction=-1)
                upopt = uncertainties.correlated_values(popt, pcov)

                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1000,
                )

                ax.plot(x_plot, self.lorentzian(x_plot, *popt), 'r-', label=rf'{i}Fit')
                annotation_text = (
                    f'Center frequency: ${upopt[0]:SL}$ MHz\n'
                    f'Width: ${1e+3 * upopt[1]:SL}$ kHz'
                )
                ax.annotate(
                    annotation_text,
                    xy=(0.02, 0.05),  # Changed to bottom-left corner (x=0.02, y=0.05)
                    xycoords='axes fraction',
                    fontsize=9,
                    ha='left',
                    va='bottom',
                )
                print(popt[0], pcov[0][0]) # print out value for plotting
                ax.legend(loc='upper right')

        fig.supxlabel(self.params[0].axis_label)
        fig.supylabel('Population')

        # fig.suptitle("Rabi Oscillation Fits", fontsize=14)

        fig.savefig(f"{self.folder_path}/grouped_survival_rate_by_site_1d.pdf")
        fig.suptitle(f"{self.folder_path}")
