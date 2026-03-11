from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Sequence, Union

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties.unumpy as unp
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from numpy.typing import NDArray
from scipy.special import comb
from uncertainties import ufloat

from analysislib.common.plot_config import PlotConfig
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.typing import StrPath


@pd.api.extensions.register_series_accessor("tools")
class LevelTools:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def split_level_custom(self, level, func, new_names=None, inplace=False):
        """
        Splits a level using a function that returns a fixed-length tuple.
        """
        target = self._obj if inplace else self._obj.copy()
        df_idx = target.index.to_frame(index=False)

        # 1. Locate the target level
        loc = df_idx.columns.get_loc(level) if isinstance(level, str) else level
        level_name = df_idx.columns[loc]

        # 2. Apply the function and expand tuples into columns
        # result_type='expand' turns the list of tuples into a DataFrame
        expanded_levels = df_idx[level_name].apply(func).apply(pd.Series)

        # 3. Handle naming
        if new_names:
            expanded_levels.columns = new_names
        else:
            expanded_levels.columns = [f"{level_name}_{i}" for i in range(len(expanded_levels.columns))]

        # 4. Reconstruct the index DataFrame
        # Remove old, insert new ones at the same position
        pre = df_idx.iloc[:, :loc]
        post = df_idx.iloc[:, loc + 1:]
        df_idx = pd.concat([pre, expanded_levels, post], axis=1)

        target.index = pd.MultiIndex.from_frame(df_idx)

        if not inplace:
            return target
        return None

# not really used, deprecate?
@pd.api.extensions.register_dataframe_accessor("tools")
class DataFrameTools:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def groupby_without_join(self, level, df_other):
        """
        Performs a groupby on the DataFrame without joining with another DataFrame.
        The grouping is done based on the values in the specified level of the index.
        """
        # Get the unique values in the specified level of the index
        level_values = self._obj.index.get_level_values(level)
        return self._obj.groupby(level_values.map(df_other))

class TweezerCorrelator(TweezerStatistician):
    KEY_POLYMER_ID: ClassVar[str] = 'polymer_id'
    KEY_POLYMER_SITE: ClassVar[str] = 'polymer_site'
    KEY_POLYMER_SURVIVAL: ClassVar[str] = 'polymer_survival'

    CMAP_MAGNETIZATION_DARK: ClassVar[Colormap] = sns.diverging_palette(
        145, 300, s=60, center='dark', as_cmap=True
    )
    CMAP_MAGNETIZATION_LIGHT: ClassVar[Colormap] = sns.diverging_palette(
        145, 300, s=60, center='light', as_cmap=True
    )

    DEFAULT_CORRELATION_CMAP: ClassVar[str] = 'coolwarm'

    def __init__(
            self,
            preproc_h5_path: StrPath,
            require_exact_rearrangement: bool,
            shot_h5_path = None,
            plot_config = None,
            *,
            polymers: Optional[Sequence[Sequence[int]]] = None,
            parity_selection: Literal[0, 1, None] = None,
            shot_index = -1,
    ):
        super().__init__(
            preproc_h5_path,
            shot_h5_path,
            plot_config,
            shot_index=shot_index,
        )
        self.require_exact_rearrangement = require_exact_rearrangement
        self.parity_selection = parity_selection

        if polymers is not None:
            self.polymers = np.asarray(polymers, dtype=int)
            if self.polymers.ndim != 2:
                raise ValueError('polymers must be 2D')
            return

        user_supplied_path = Path(preproc_h5_path)
        if user_supplied_path.is_dir():
            h5_path = user_supplied_path / TweezerPreprocessor.PROCESSED_RESULTS_FNAME
        else:
            h5_path = user_supplied_path

        with h5py.File(h5_path, 'r') as f:
            self.polymers = np.asarray(f.attrs['target_array'][:], dtype=int)
            if self.polymers.ndim != 2:
                raise ValueError(
                    'TW_target_array was not 2D as expected for polymer grouping. '
                    'Please ensure TW_target_array is a 2D array of shape (n_polymers, polymer_length).'
                )
            self.target_sites = self.polymers.flatten()

    def _polymer_grouper(self, sites):
        # binary search on the flattened view of self.polymers
        flat_inds = np.searchsorted(self.polymers.ravel(), np.asarray(sites))

        # convert 1D indices to 2D (row, col)
        # np.divmod returns both quotient (row) and remainder (column)
        return np.divmod(flat_inds, self.polymer_length)

    @property
    def n_polymers(self):
        return self.polymers.shape[0]

    @property
    def polymer_length(self):
        return self.polymers.shape[1]

    def polymer_survivals(self) -> pd.Series:
        """
        Return a Series of booleans indicating survival of target sites,
        hierarchically indexed by shot, polymer, and site within polymer.
        Keys are given by class variables KEY_POLYMER_ID and KEY_POLYMER_SITE.

        If self.require_exact_rearrangement is True, only shots with perfect rearrangement (no missing or spurious atoms) are included.

        Example
        -------

        ```
        shot  polymer_id  polymer_site
        0     0           0               False
                          1               False
                          2               False
              1           0               False
                          1               False
                                          ...
        2496  2           1               False
                          2               False
              3           0                True
                          1               False
                          2               False
        Name: occupancy, Length: 14580, dtype: bool
        ```
        """
        series = self.series()

        if self.require_exact_rearrangement:
            # occupancies on image 1
            rearranged_unstack = series.xs(1, level=self.KEY_IMAGE).unstack()
            """
            site      0      1      2     3      4      5
            shot
            0     False  False  False  True  False  False
            1     False  False  False  True  False  False
            2     False  False  False  True  False  False
            3     False  False  False  True  False  False
            4     False  False  False  True  False  False
            5     False  False  False  True  False  False
            """

            # index consisting of only perfectly rearranged arrays (no missing or spurious atoms)
            filtered_indices = rearranged_unstack.index[(rearranged_unstack == self.target_sites_mask).all(axis=1)]
            """
            Index([   0,    1,    2,    3,    8,   10,   13,   14,   15,   16,
                   ...
                   2485, 2486, 2487, 2488, 2490, 2491, 2492, 2494, 2495, 2496],
                  dtype='int64', name='shot', length=1215)
            """

            survivals_masked = series.xs(2, level=self.KEY_IMAGE).loc[filtered_indices, self.target_sites]
        else:
            survivals_masked = series.xs(2, level=self.KEY_IMAGE).loc[:, self.target_sites]

        return survivals_masked.tools.split_level_custom(
            'site',
            self._polymer_grouper,
            new_names=[self.KEY_POLYMER_ID, self.KEY_POLYMER_SITE],
        )

    def polymer_total_survival(self):
        """
        Return a Series of total survivals (number of surviving sites) per polymer,
        indexed by shot and polymer_id.

        Example
        -------
        ```
        shot  polymer_id
        0     0             0
              1             0
              2             1
              3             1
        1     0             1
                           ..
        2495  3             1
        2496  0             2
              1             2
              2             0
              3             1
        Name: total_survival, Length: 4860, dtype: int64
        ```
        """
        return self.polymer_survivals() \
            .unstack() \
            .sum(axis=1) \
            .rename(self.KEY_POLYMER_SURVIVAL, inplace=True)

    @staticmethod
    def _bools_to_bitstring(bools: Sequence[bool]) -> str:
        """
        Convert a sequence of booleans to a bitstring (e.g., [True, False, True] -> '101').

        Parameters
        ----------
        bools : Sequence[bool]
            Sequence of boolean values.

        Returns
        -------
        str
            Bitstring representation of the boolean sequence.

        Examples
        --------
        >>> TweezerCorrelator._bools_to_bitstring([True, False, True])
        '101'

        >>> TweezerCorrelator._bools_to_bitstring([False, False, True, True])
        '0011'
        """
        return ''.join(str(int(b)) for b in bools)

    def bitstrings(self) -> pd.DataFrame:
        """
        Give the final bitstring of each polymer.
        Will be post-selected for exact rearrangement,
        if so specified at object instantiation.

        Example
        -------
        >>> tc = TweezerCorrelator(...)
        >>> tc.bitstrings()

        shot  polymer_id
        0     0             000
              1             000
              2             010
              3             100
        1     0             010
                           ...
        2495  3             010
        2496  0             101
              1             110
              2             000
              3             100
        Name: bitstring, Length: 4860, dtype: object
        """
        bitstrings = self.polymer_survivals() \
            .unstack() \
            .agg(self._bools_to_bitstring, axis=1)
        bitstrings.rename('bitstring', inplace=True)
        return bitstrings

    @staticmethod
    def _normalize_with_laplace_errors(counts: pd.Series) -> pd.Series:
        """
        Normalize a series of categorical counts
        and compute binomial errors with Laplace correction.
        Intended for use in DataFrame.apply().

        Parameters
        ----------
        counts : pd.Series
            Series of counts for each category (e.g., bitstring).

        Returns
        -------
        pd.Series
            Series of uarrays with normalized frequencies and Laplace error bars.

        Examples
        --------
        >>> counts = pd.Series({'00': 10, '01': 20, '10': 30, '11': 40})
        >>> TweezerCorrelator._normalize_with_laplace_errors(counts)

        00    0.100+/-0.031
        01      0.20+/-0.04
        10      0.30+/-0.05
        11      0.40+/-0.05
        dtype: object
        """
        n = counts.sum()
        p_laplace = (counts + 1) / (n + 2)
        errors = np.sqrt(p_laplace * (1 - p_laplace) / (n + 2))
        normalized = counts / n
        return pd.Series(unp.uarray(normalized.values, errors.values), index=counts.index)

    def bitstring_frequencies(self, grouped_by: Sequence[str] = []) -> pd.DataFrame:
        """
        Compute frequencies of each bitstring (combination of survivals across polymer sites),
        grouped by specified parameters.

        Parameters
        ----------
        grouped_by : list of str
            List of parameter names to group by (e.g., ['polymer_id']).

        Returns
        -------
        DataFrame of uarrays with index matching groupby dimensions.

        Example
        -------
        >>> tc = TweezerCorrelator(...)
        >>> tc.bitstring_frequencies()

        bitstring                          000            001            010
        mmwave_ramsey_wait_time
        0.000000e+00             0.723+/-0.033  0.085+/-0.021  0.043+/-0.015
        1.458333e-07             0.714+/-0.030  0.100+/-0.020  0.068+/-0.017
        2.916667e-07               0.61+/-0.04  0.103+/-0.023  0.076+/-0.020
        4.375000e-07               0.55+/-0.04  0.083+/-0.021  0.083+/-0.021
        5.833333e-07             0.439+/-0.035  0.082+/-0.020  0.061+/-0.018

        >>> tc.bitstring_frequencies(grouped_by=['polymer_id'])

        bitstring                                   000            001            010
        polymer_id mmwave_ramsey_wait_time
        0          0.000000e+00             0.81+/-0.06  0.021+/-0.028  0.043+/-0.034
                   1.458333e-07             0.71+/-0.06    0.13+/-0.05    0.07+/-0.04
                   2.916667e-07             0.70+/-0.07    0.09+/-0.04    0.07+/-0.04
                   4.375000e-07             0.49+/-0.07    0.07+/-0.04    0.13+/-0.05
                   5.833333e-07             0.41+/-0.07    0.12+/-0.05  0.020+/-0.027
        ...                                         ...            ...            ...
        3          2.916667e-06             0.39+/-0.07    0.09+/-0.05    0.20+/-0.06
                   3.062500e-06             0.41+/-0.07    0.09+/-0.05    0.11+/-0.05
                   3.208333e-06             0.39+/-0.08    0.13+/-0.06    0.11+/-0.05
                   3.354167e-06             0.32+/-0.07    0.12+/-0.05    0.05+/-0.04
                   3.500000e-06             0.39+/-0.07    0.13+/-0.05  0.019+/-0.025
        """
        bitstrings = self.bitstrings()

        scan_params = self.scan_param_df()
        bitstrings_filtered = bitstrings[self._parity_filter()]
        bitstring_totals = bitstrings_filtered.to_frame() \
            .join(scan_params, on=self.KEY_SHOT) \
            .groupby(list(grouped_by) + scan_params.columns.tolist()) \
            .value_counts() \
            .unstack(fill_value=0)

        bitstring_freqs = bitstring_totals.apply(
            self._normalize_with_laplace_errors,
            axis=1,
            result_type='expand',
        )

        return bitstring_freqs

    def distance_averaged_correlation(self, grouped_by=[]):
        """
        The correlation function <Z_i Z_{i+r}> - <Z_i><Z_{i+r}> averaged over all i.
        (That is, the average of the covariance matrix along diagonals.)
        Here, Z_i refers to a Pauli operator with value +/-1.
        """
        scan_params = self.scan_param_df()

        grouping = grouped_by + scan_params.columns.tolist()
        survivals_unstacked_filt = self.polymer_survivals().unstack()[self._parity_filter()]
        corrs = (2 * survivals_unstacked_filt) \
            .join(scan_params, on=self.KEY_SHOT) \
            .groupby(grouping) \
            .cov(ddof=1) \
            .stack()
        # corrs.index = corrs.index.set_names([self.KEY_POLYMER_SITE + '_1', self.KEY_POLYMER_SITE + '_2'], level=[-2, -1])

        distances = corrs.index.get_level_values(-2) - corrs.index.get_level_values(-1)
        distance_avgd_corrs = corrs.groupby(grouping + [distances]).mean()
        distance_avgd_corrs.index = distance_avgd_corrs.index.set_names('distance', level=-1)
        return distance_avgd_corrs

    def _parity_filter(self):
        if self.parity_selection is not None:
            total_survivals = self.polymer_total_survival()
            return total_survivals % 2 == self.parity_selection
        else:
            return slice(None)

    def _compute_total_survivals_ufreqs(self, grouped_by: Sequence[str] = []) -> pd.DataFrame:
        """
        Compute normalized frequencies for total survivals (i.e. z-magnetization) of each polymer,
        grouped by scan parameters and optionally further grouped by polymer_id, with Laplace errorbars.

        Parameters
        ----------
        grouped_by : Sequence[str]
            List of parameter names to group by (e.g., ['polymer_id']).

        Returns
        -------
        DataFrame of uarrays with index matching groupby dimensions.
        """
        total_survivals = self.polymer_total_survival()
        filtered_survivals = total_survivals[self._parity_filter()]

        scan_params = self.scan_param_df()
        group_cols = list(grouped_by) + scan_params.columns.to_list()

        survivals_counts = filtered_survivals.to_frame() \
            .join(scan_params, on='shot') \
            .groupby(group_cols) \
            .value_counts() \
            .unstack(fill_value=0)

        return survivals_counts.apply(
            self._normalize_with_laplace_errors,
            axis=1,
            result_type='expand',
        )

    def _plot_ufreqs(
            self,
            ax: Axes,
            ufreqs_data,
            label_mapper: Callable[[Any], str] = str,
            color_mapper: Callable[[Any], Optional[ColorType]] = lambda _: None
    ) -> None:
        """Helper to plot frequency data (with errorbars) on a single axis.

        Parameters
        ----------
        ax : Axes
            Axis to plot on.
        ufreqs_data : Series or DataFrame
            Data with errorbar values (uarray).
        cmap : str or Colormap, optional
            Colormap for different magnetization levels. If unspecified, a default diverging palette is used.
        """

        subplotspec = ax.get_subplotspec()
        if subplotspec is None:
            raise ValueError("Axis must be part of a subplot to determine x-axis labeling.")

        variable_scaled, xlabel, xscale = self._scale_independent_variable(ufreqs_data.index)
        for column_name, column in ufreqs_data.items():
            ax.errorbar(
                variable_scaled,
                unp.nominal_values(column.values),
                yerr=unp.std_devs(column.values),
                label=label_mapper(column_name),
                color=color_mapper(column_name),
                **self.plot_config.errorbar_kw,
            )
        if subplotspec.is_last_row():
            ax.set_xlabel(xlabel)

    def plot_magnetization_pops(
            self,
            axs: Optional[Axes | Sequence[Axes]] = None,
            cmap: Optional[str | matplotlib.colors.Colormap] = None,
    ):
        """Plot population in various magnetization states vs scan parameters.

        Parameters
        ----------
        axs : Axes or Sequence[Axes], optional
            If Axes: aggregate plot on single axes.
            If Sequence[Axes]: per-polymer plots on axs[polymer_id].
            If None: creates new figure with single axes.
        """
        if axs is None:
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplots()

        def label_mapper(survival_number: int) -> str:
            # return f'$S_z = {survival_number - self.polymer_length/2:+}$'
            return f'{survival_number} survivors'

        colormap = cmap
        if cmap is None:
            colormap = self.CMAP_MAGNETIZATION_DARK

        def color_mapper(column_name):
            norm = matplotlib.colors.Normalize(vmin=0, vmax=self.polymer_length)
            return colormap(norm(column_name))

        if isinstance(axs, Axes):
            ax = axs
            # Single axis mode: aggregate
            ufreqs = self._compute_total_survivals_ufreqs()
            self._plot_ufreqs(ax, ufreqs, label_mapper=label_mapper, color_mapper=color_mapper)
            ax.set_ylabel('Population')
            ax.legend()
        else:
            # Sequence of axes: per-polymer
            ufreqs = self._compute_total_survivals_ufreqs(grouped_by=[self.KEY_POLYMER_ID])
            for polymer_id, group in ufreqs.groupby(level=self.KEY_POLYMER_ID):
                ax = axs[polymer_id]
                group_data = group.droplevel(self.KEY_POLYMER_ID)
                self._plot_ufreqs(ax, group_data, label_mapper=label_mapper, color_mapper=color_mapper)
                ax.set_ylabel(f'Polymer {polymer_id} population')
            axs[0].legend()

    @staticmethod
    def _mean_with_std_err(series: pd.Series):
        return ufloat(series.mean(), series.std(ddof=1) / np.sqrt(series.count()))

    def plot_parity(self, ax: Axes):
        # +/-1 for even/odd number of survivals
        polymer_parities = (-1) ** (self.polymer_total_survival() % 2)

        scan_params = self.scan_param_df()
        mean_parities: pd.Series = polymer_parities.to_frame() \
            .join(scan_params, on='shot') \
            .groupby(scan_params.columns.to_list()) \
            .agg(self._mean_with_std_err) \
            .iloc[:, 0]

        ax.errorbar(
            mean_parities.index,
            unp.nominal_values(mean_parities),
            yerr=unp.std_devs(mean_parities),
            **self.plot_config.errorbar_kw,
        )

        ax.set_ylabel('Parity')

    def _plot_heatmap_on_axis(
            self,
            ax: Axes,
            data,
            ylabel: str,
            cmap: Union[str, Colormap] = 'viridis',
            vmin: float = 0,
            vmax: float = 1,
            title: Optional[str] = None,
    ):
        """Helper to plot heatmap data on a single axis.

        Parameters
        ----------
        ax : Axes
            Axis to plot on.
        data : DataFrame
            Data with columns as x-axis and index as y-axis.
        ylabel : str
            Label for y-axis.
        cmap : str
            Colormap name.
        vmin, vmax : float
            Color scale limits.
        title : str, optional
            Title for the axis.
        """

        x_scaled, xlabel, _ = self._scale_independent_variable(data.index)
        pcmesh = ax.pcolormesh(
            x_scaled,
            data.columns,
            unp.nominal_values(data.T),
            shading='nearest',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        subplotspec = ax.get_subplotspec()
        if subplotspec is None:
            raise ValueError("Axis must be part of a subplot to determine x-axis labeling.")
        if subplotspec.is_last_row():
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig = ax.figure
        if fig is None:
            raise ValueError("Axis must be part of a figure to add colorbar.")
        fig.colorbar(pcmesh, ax=ax)

    def plot_local_magnetization(self, ax: Optional[Axes] = None):
        scan_params = self.scan_param_df()
        occupancies_with_params = self.polymer_survivals() \
            .to_frame() \
            .join(scan_params, on='shot')
        """
                                      occupancy  mmwave_ramsey_wait_time
        shot polymer_id polymer_site
        0    0          0                 False                 0.000000
                        1                 False                 0.000000
             1          0                 False                 0.000000
                        1                 False                 0.000000
             2          0                 False                 0.000000
        ...                                 ...                      ...
        1499 0          1                 False                 0.000006
             1          0                 False                 0.000006
                        1                 False                 0.000006
             2          0                  True                 0.000006
                        1                 False                 0.000006
        """

        local_mags = occupancies_with_params.groupby(scan_params.columns.to_list() + ['polymer_site']) \
            .mean()['occupancy'] \
            .unstack()

        if ax is None:
            fig = plt.figure(constrained_layout=True)
            ax = fig.subplots()
        self._plot_heatmap_on_axis(
            ax,
            local_mags,
            ylabel='Polymer site',
            cmap=self.CMAP_MAGNETIZATION_LIGHT,
            vmin=0,
            vmax=1,
        )

    def plot_bitstring_populations(self, axs: Optional[Axes | Sequence[Axes]] = None):
        """Plot bitstring populations.

        Parameters
        ----------
        axs : Axes or Sequence[Axes], optional
            If Axes: aggregate plot on single axes.
            If Sequence[Axes]: per-polymer plots on axs[polymer_id].
            If None: creates new figure with single axes.
        """
        if axs is None:
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplots()

        if isinstance(axs, Axes):
            ax = axs
            # Single axis mode: aggregate
            bitstring_freqs = self.bitstring_frequencies()
            self._plot_ufreqs(ax, bitstring_freqs)
            ax.set_ylabel('Population')
            ax.legend()
            ax.grid()
        else:
            # Sequence of axes: per-polymer
            bitstring_freqs = self.bitstring_frequencies(grouped_by=[self.KEY_POLYMER_ID])
            for polymer_id, group in bitstring_freqs.groupby(level=self.KEY_POLYMER_ID):
                ax = axs[polymer_id]
                group_data = group.droplevel(self.KEY_POLYMER_ID)
                self._plot_ufreqs(ax, group_data)
                ax.set_ylabel('Population')
                ax.legend()
                ax.grid()

    def plot_bitstring_heatmap(self, axs: Optional[Axes | Sequence[Axes]] = None):
        """Plot bitstring frequency heatmap.

        Parameters
        ----------
        axs : Axes or Sequence[Axes], optional
            If Axes: aggregate plot on single axes.
            If Sequence[Axes]: per-polymer plots on axs[polymer_id].
            If None: creates new figure with single axes.
        """
        if axs is None:
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplots()

        if isinstance(axs, Axes):
            # Single axis mode: aggregate
            bitstring_freqs = self.bitstring_frequencies()
            self._plot_heatmap_on_axis(axs, bitstring_freqs, ylabel='Bitstring', cmap='inferno')
        else:
            # Sequence of axes: per-polymer
            bitstring_freqs = self.bitstring_frequencies(grouped_by=[self.KEY_POLYMER_ID])
            for polymer_id, group in bitstring_freqs.groupby(level=self.KEY_POLYMER_ID):
                ax = axs[polymer_id]
                group_data = group.droplevel(self.KEY_POLYMER_ID)
                self._plot_heatmap_on_axis(ax, group_data, ylabel='Bitstring', cmap='inferno', title=f'Polymer {polymer_id}')

    def plot_distance_averaged_correlation(self, axs: Optional[Axes | Sequence[Axes]] = None):
        """Plot distance-averaged correlation heatmap.

        Parameters
        ----------
        axs : Axes or Sequence[Axes], optional
            If Axes: aggregate plot on single axes.
            If Sequence[Axes]: per-polymer plots on axs[polymer_id].
            If None: creates new figure with single axes.
        """
        if axs is None:
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplots()

        if isinstance(axs, Axes):
            # Single axis mode: aggregate
            distance_avgd_corrs = self.distance_averaged_correlation().unstack()
            distance_avgd_corrs.loc(axis=1)[0] = np.nan
            self._plot_heatmap_on_axis(axs, distance_avgd_corrs, ylabel='Distance (sites)', cmap='coolwarm', vmin=-1, vmax=1)
        else:
            # Sequence of axes: per-polymer - compute correlation per polymer
            distance_avgd_corrs = self.distance_averaged_correlation(grouped_by=[self.KEY_POLYMER_ID]).unstack()
            for polymer_id, group in distance_avgd_corrs.groupby(level=self.KEY_POLYMER_ID):
                ax = axs[polymer_id]
                group_data = group.droplevel(self.KEY_POLYMER_ID)
                group_data.loc(axis=1)[0] = np.nan
                self._plot_heatmap_on_axis(ax, group_data, ylabel='Distance (sites)', cmap='coolwarm', vmin=-1, vmax=1, title=f'Polymer {polymer_id}')
