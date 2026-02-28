from __future__ import annotations

from typing import ClassVar, Literal, Optional, Sequence

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties.unumpy as unp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.special import comb

from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig
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

    def __init__(
            self,
            preproc_h5_path,
            require_exact_rearrangement: bool,
            shot_h5_path = None,
            plot_config = None,
            *,
            polymers: Optional[Sequence[Sequence[int]]] = None,
            parity_selection: Literal[0, 1, None] = None,
            shot_index = -1,
    ):
        if polymers is None:
            target_sites = np.array([], dtype=int)
        else:
            target_sites = np.asarray(polymers).flatten()

        super().__init__(
            preproc_h5_path,
            shot_h5_path,
            plot_config,
            shot_index=shot_index,
            target_sites=target_sites,  # remove in 2 wks after 2026-02-27
        )
        if not self.rearrangement:
            raise ValueError('TweezerCorrelator can only be run on shots with rearrangement')
        self.require_exact_rearrangement = require_exact_rearrangement
        self.parity_selection = parity_selection

        with h5py.File(preproc_h5_path, 'r') as f:
            try:
                self.polymers = np.asarray(f.attrs['target_array'][:], dtype=int)
            except KeyError:
                # deprecate this soon
                self.polymers = np.asarray(polymers, dtype=int)
            if self.polymers.ndim != 2:
                raise ValueError(
                    'TW_target_array was not 2D as expected for polymer grouping. '
                    'Please ensure TW_target_array is a 2D array of shape (n_polymers, polymer_length).'
                )

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
        bitstrings = self.polymer_survivals() \
            .unstack() \
            .agg(self._bools_to_bitstring, axis=1) \
            .rename('bitstring', inplace=True)
        """
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

        scan_params = self.scan_param_df()
        bitstrings_filtered = bitstrings[self._parity_filter()]
        bitstring_totals = bitstrings_filtered.to_frame() \
            .join(scan_params, on=self.KEY_SHOT) \
            .groupby(grouped_by + scan_params.columns.tolist()) \
            .value_counts() \
            .unstack(fill_value=0)

        bitstring_freqs = bitstring_totals.apply(
            self._normalize_with_laplace_errors,
            axis=1,
            result_type='expand',
        )

        return bitstring_freqs

    def distance_averaged_correlation(self, grouped_by=[]):
        scan_params = self.scan_param_df()

        grouping = grouped_by + scan_params.columns.tolist()
        survivals_unstacked_filt = self.polymer_survivals().unstack()[self._parity_filter()]
        corrs = survivals_unstacked_filt \
            .join(scan_params, on=self.KEY_SHOT) \
            .groupby(grouping) \
            .corr() \
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
        group_cols = grouped_by + scan_params.columns.to_list()

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
    
    def _plot_magnetization_on_axis(self, ax: Axes, ufreqs_data, cmap=None, title: Optional[str] = None):
        """Helper to plot magnetization data on a single axis.
        
        Parameters
        ----------
        ax : Axes
            Axis to plot on.
        ufreqs_data : Series or DataFrame
            Data with errorbar values (uarray).
        cmap : str or Colormap, optional
            Colormap for different magnetization levels. If unspecified, a default diverging palette is used.
        title : str, optional
            Title for the axis.
        """
        colormap = cmap
        if cmap is None:
            colormap = sns.diverging_palette(145, 300, s=60, center='dark', as_cmap=True)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.polymer_length)
        for column_name, column in ufreqs_data.items():
            ax.errorbar(
                column.index,
                unp.nominal_values(column.values),
                yerr=unp.std_devs(column.values),
                label=f'$S_z = {column_name - self.polymer_length/2:+}$',
                color=colormap(norm(column_name)),
                **self.plot_config.errorbar_kw,
            )
        if title:
            ax.set_title(title)
        ax.legend()
    
    def plot_total_magnetization(
            self,
            axs: Optional[Axes | Sequence[Axes]] = None,
            cmap: Optional[str | matplotlib.colors.Colormap] = None,
    ):
        """Plot total magnetization survivals vs scan parameters.
        
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
            ufreqs = self._compute_total_survivals_ufreqs()
            self._plot_magnetization_on_axis(axs, ufreqs, cmap=cmap)
        else:
            # Sequence of axes: per-polymer
            ufreqs = self._compute_total_survivals_ufreqs(grouped_by=[self.KEY_POLYMER_ID])
            for polymer_id, group in ufreqs.groupby(level=self.KEY_POLYMER_ID):
                ax = axs[polymer_id]
                group_data = group.droplevel(self.KEY_POLYMER_ID)
                self._plot_magnetization_on_axis(ax, group_data, cmap=cmap, title=f'Polymer {polymer_id}')

    def _plot_heatmap_on_axis(self, ax: Axes, data, ylabel: str, cmap: str = 'viridis', vmin: float = 0, vmax: float = 1, title: Optional[str] = None):
        """Helper to plot heatmap data on a single axis.
        
        Parameters
        ----------
        ax : Axes
            Axis to plot on.
        data : DataFrame
            Data with columns as x-axis and index as y-axis (should be pre-unstacked).
        ylabel : str
            Label for y-axis.
        cmap : str
            Colormap name.
        vmin, vmax : float
            Color scale limits.
        title : str, optional
            Title for the axis.
        """
        pcmesh = ax.pcolormesh(
            data.index,
            data.columns,
            unp.nominal_values(data.T),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig = ax.figure
        fig.colorbar(pcmesh, ax=ax)

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

class TweezerCorrelatorVibed(TweezerStatistician):
    """Analyze multi-site correlations using bitstring basis {0,1}^n.
    
    Extends TweezerStatistician for multi-site correlation analysis beyond pairs.
    Automatically uses target_sites as selected_sites in rearrangement mode.
    
    Parametersall 
    ----------
    preproc_h5_path : str
        Path to the processed quantities h5 file
    target_sites : Sequence[int]
        Sites to analyze (used as rearrangement targets)
    shot_h5_path : str, optional
        Path to the shot h5 file
    plot_config : PlotConfig, optional
        Configuration object for plot styling
    shot_index : int, default=-1
        Shot index to analyze
    """
    
    selected_sites: Sequence[int]
    '''Indices of sites selected for correlation analysis (same as target_sites)'''
    
    def __init__(
            self,
            preproc_h5_path: StrPath,
            target_sites: Sequence[int],
            shot_h5_path: Optional[StrPath] = None,
            plot_config: Optional[PlotConfig] = None,
            *,
            shot_index: int = -1,
    ):
        """Initialize with target sites for rearrangement analysis."""
        super().__init__(
            preproc_h5_path=preproc_h5_path,
            shot_h5_path=shot_h5_path,
            plot_config=plot_config,
            rearrangement=True,
            shot_index=shot_index,
            target_sites=target_sites,
        )
        # selected_sites = target_sites for correlation analysis
        self.selected_sites = target_sites
        
        # Validate selected sites
        if len(self.selected_sites) == 0:
            raise ValueError("target_sites must contain at least one site")
        if max(self.selected_sites) >= self.n_sites:
            raise ValueError(
                f"target_sites contains indices >= n_sites ({self.n_sites})"
            )
    
    def extract_bitstrings(
            self,
            image_index: int = -1,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
    ) -> NDArray:
        """Extract survival bitstrings for selected sites (site[0] is the most-significant bit).
        
        shot_mask is element-wise data applied uniformly within each shot.
        It filters which elements to consider across all shots.
        Returns shape (shots, n_selected_elements).
        """

        # Use parent class's surviving_atoms_array for correct image indices
        survival_all_sites = self.surviving_atoms_array  # shape: (shots, sites)
        
        # Extract survival for selected sites only
        if require_exact_rearrangement:
            exact_mask = self._shot_mask_exact_rearrangement()
            occupancies = survival_all_sites[exact_mask, :]
            selected_occupancies = occupancies[:, self.selected_sites]
        else:
            selected_occupancies = survival_all_sites[:, self.selected_sites]
        
        # Combine rearrangement mask with any additional shot mask
        combined_mask = shot_mask
        
        # Apply combined mask(s)
        if combined_mask is not None:
            # Check if multiple masks provided
            if isinstance(combined_mask, (list, tuple)):
                # Multiple masks: apply each mask element-wise and return as separate arrays
                # For averaging later, we'll average at the population counting level
                bitstrings_list = []
                for mask in combined_mask:
                    masked_occupancies = selected_occupancies[:, mask]  # Apply element-wise mask
                    bitstrings_list.append(masked_occupancies)
                return bitstrings_list  # Return list of (shots, n_elements) arrays
            else:
                # Single mask: apply element-wise mask
                selected_occupancies = selected_occupancies[:, combined_mask]  # Apply element-wise mask
        
        return selected_occupancies.astype(int)
    
    def _bitstring_to_tuple(self, bitstring_row: NDArray) -> tuple[int, ...]:
        """Convert bitstring array to hashable tuple."""
        return tuple(int(x) for x in bitstring_row)
    
    def _tuple_to_bitstring(self, bitstring_tuple: tuple[int, ...]) -> str:
        """Convert bitstring tuple to string (e.g., (1,0,1) -> '101')."""
        return ''.join(str(int(x)) for x in bitstring_tuple)
    
    def bitstring_populations(
            self,
            image_index: int = -1,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
    ) -> dict[tuple, float]:
        """Return dict of bitstring populations, averaged over masks/reps."""
        from itertools import product
        n_sites = len(self.selected_sites)
        
        # Combine rearrangement mask with any additional shot mask
        combined_mask = shot_mask
        
        # Handle multiple masks by computing populations separately and averaging
        if isinstance(combined_mask, (list, tuple)):
            populations_list = []
            for mask in combined_mask:
                bitstrings = self.extract_bitstrings(image_index=image_index, shot_mask=None, require_exact_rearrangement=False)
                bitstrings = bitstrings[mask]  # Apply mask manually
                
                # Convert each bitstring row to tuple for use as dict key
                bitstring_tuples = [self._bitstring_to_tuple(bs) for bs in bitstrings]
                
                # Count occurrences
                pops = {}
                for bs_tuple in set(bitstring_tuples):
                    count = bitstring_tuples.count(bs_tuple)
                    pops[bs_tuple] = float(count / len(bitstring_tuples))
                
                # Ensure all possible bitstrings are represented
                for bs_tuple in product([0, 1], repeat=n_sites):
                    if bs_tuple not in pops:
                        pops[bs_tuple] = 0.0
                
                populations_list.append(pops)
            
            # Average populations across all masks
            populations = {}
            for bs_tuple in product([0, 1], repeat=n_sites):
                populations[bs_tuple] = float(np.mean([pops[bs_tuple] for pops in populations_list]))
            
            return populations
        else:
            # Single mask (combined rearrangement + optional filter)
            bitstrings = self.extract_bitstrings(image_index=image_index, shot_mask=shot_mask, require_exact_rearrangement=require_exact_rearrangement)
            
            # Convert each bitstring row to tuple for use as dict key
            bitstring_tuples = [self._bitstring_to_tuple(bs) for bs in bitstrings]
            
            # Count occurrences
            populations = {}
            for bs_tuple in set(bitstring_tuples):
                count = bitstring_tuples.count(bs_tuple)
                populations[bs_tuple] = float(count / len(bitstring_tuples))
            
            # Ensure all possible bitstrings are represented (with 0 population if absent)
            for bs_tuple in product([0, 1], repeat=n_sites):
                if bs_tuple not in populations:
                    populations[bs_tuple] = 0.0
            
            return populations
    
    def _validate_shot_masks(self, shot_mask: Sequence[np.ndarray]):
        """Validate that all masks have the same shape for averaging."""
        if not isinstance(shot_mask, (list, tuple)) or len(shot_mask) <= 1:
            return
        
        first_shape = shot_mask[0].shape
        for i, mask in enumerate(shot_mask[1:], start=1):
            if mask.shape != first_shape:
                raise ValueError(
                    f"All shot masks must have the same shape for averaging. "
                    f"Mask 0 has shape {first_shape}, but mask {i} has shape {mask.shape}."
                )
    
    def plot_bitstring_populations_heatmap(
            self,
            ax: Optional[Axes] = None,
            fig: Optional[Figure] = None,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
            average_shot_masks: bool = True,
            save_fig = True,
    ):
        """Heatmap of bitstring populations vs parameter (lexicographic order).
        
        X-axis: unique parameter values, Y-axis: bitstrings, Color: population.
        
        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on (ignored if multiple masks and average_shot_masks=False).
        fig : Figure, optional
            Figure to plot on (creates new if None and ax is None).
        shot_mask : np.ndarray or Sequence[np.ndarray], optional
            Additional mask(s) to filter shots. Multiple masks can be averaged
            or plotted separately.
        require_exact_rearrangement : bool, default=True
            Apply rearrangement mask based on target_sites at image 1.
        average_shot_masks : bool, default=True
            If True, average multiple masks. If False, create separate subplots.
        """
        # Validate masks if averaging
        if average_shot_masks:
            self._validate_shot_masks(shot_mask)
        
        has_multiple = (isinstance(shot_mask, (list, tuple)) and len(shot_mask) > 1) or \
                       (isinstance(shot_mask, np.ndarray) and shot_mask.ndim == 2 and shot_mask.shape[0] > 1)
        plot_separately = has_multiple and not average_shot_masks
        
        if plot_separately:
            # Create subplots for each mask
            n_masks = len(shot_mask)
            if fig is None:
                fig = plt.figure(
                    figsize=(self.plot_config.figure_size[0], 
                             self.plot_config.figure_size[1] * n_masks),
                    constrained_layout=self.plot_config.constrained_layout,
                )
            axes = fig.subplots(nrows=1, ncols=n_masks, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            
            # Handle both list/tuple and 2D array forms of multiple masks
            masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                         [shot_mask[i] for i in range(shot_mask.shape[0])]
            for mask_idx, mask in enumerate(masks_iter):
                self._plot_heatmap_single_mask(
                    ax=axes[mask_idx],
                    shot_mask=mask,
                    require_exact_rearrangement=require_exact_rearrangement,
                    # title_suffix=f" (Mask {mask_idx+1})"
                )
        else:
            # Single mask, no mask, or averaging multiple masks
            if ax is None:
                if fig is None:
                    fig = plt.figure(
                        figsize=self.plot_config.figure_size,
                        constrained_layout=self.plot_config.constrained_layout,
                    )
                ax = fig.subplots()
            else:
                save_fig = False
            
            self._plot_heatmap_single_mask(
                ax=ax,
                shot_mask=shot_mask,
                require_exact_rearrangement=require_exact_rearrangement,
                # title_suffix=""
            )

        fig.suptitle(str(self.folder_path), fontsize=8)

        plt.tight_layout()
            
        if save_fig:
            figname = self.folder_path / 'bitstring_populations_heatmap.pdf'
            fig.savefig(figname)
            plt.show()
    
    def _plot_heatmap_single_mask(
            self,
            ax: Axes,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
            title_suffix: str = "",
    ):
        """Plot heatmap for one or more masks (averaged if multiple)."""
        # Determine if we're averaging multiple masks
        average_masks = (isinstance(shot_mask, (list, tuple))) or \
                        (isinstance(shot_mask, np.ndarray) and shot_mask.ndim == 2)
        
        # Get unique scan parameters
        unique_params_arr = self.unique_params()
        if unique_params_arr.ndim == 1:
            unique_params_arr = unique_params_arr[:, np.newaxis]
        
        n_unique = unique_params_arr.shape[0]
        
        # Determine bitstring size from the mask, not from selected_sites
        if average_masks:
            masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                         [shot_mask[i] for i in range(shot_mask.shape[0])]
            first_mask = masks_iter[0]
        else:
            first_mask = shot_mask
        
        # Get mask size
        if first_mask is None:
            n_sites = len(self.selected_sites)
        elif isinstance(first_mask, (list, np.ndarray)):
            n_sites = len(first_mask)
        else:
            # Boolean array - count True values
            n_sites = np.sum(first_mask)
        
        n_bitstrings = 2 ** n_sites
        
        from itertools import product
        all_bitstrings = [tuple(bs) for bs in product([0, 1], repeat=n_sites)]
        bitstring_to_idx = {bs: i for i, bs in enumerate(all_bitstrings)}
        populations_heatmap = np.zeros((n_bitstrings, n_unique))
        
        # For each unique parameter value
        loop_params = self._loop_params()
        if loop_params.ndim == 1:
            loop_params = loop_params[:, np.newaxis]
        
        for param_idx, param_val in enumerate(unique_params_arr):
            matches_temp = np.all(loop_params == param_val[np.newaxis, :], axis=1)
            if require_exact_rearrangement:
                exact_mask = self._shot_mask_exact_rearrangement()
                matches = matches_temp[exact_mask]
            else:
                matches = matches_temp
            if not matches.any():
                continue
            
            if average_masks:
                # Average populations across multiple masks
                pops_list = []
                # Handle both list/tuple and 2D array forms of multiple masks
                masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                             [shot_mask[i] for i in range(shot_mask.shape[0])]
                for mask in masks_iter:
                    # Extract bitstrings with element mask, then filter by parameter
                    bitstrings = self.extract_bitstrings(
                        shot_mask=mask,
                        require_exact_rearrangement=require_exact_rearrangement,
                    )
                    # extract_bitstrings returns 2D array (shots, n_elements)
                    bitstrings_for_param = bitstrings[matches]
                    if len(bitstrings_for_param) > 0:
                        pops, _ = self._compute_populations(bitstrings_for_param, all_bitstrings)
                        pops_list.append(pops)
                
                if pops_list:
                    for bs_tuple in all_bitstrings:
                        avg = np.mean([pops.get(bs_tuple, 0) for pops in pops_list])
                        populations_heatmap[bitstring_to_idx[bs_tuple], param_idx] = avg
            else:
                # Single mask or no mask
                bitstrings = self.extract_bitstrings(
                    shot_mask=shot_mask,
                    require_exact_rearrangement=require_exact_rearrangement,
                )
                # extract_bitstrings returns 2D array (shots, n_elements)
                bitstrings_for_param = bitstrings[matches]
                if len(bitstrings_for_param) > 0:
                    pops, _ = self._compute_populations(bitstrings_for_param, all_bitstrings)
                    for bs_tuple, pop in pops.items():
                        populations_heatmap[bitstring_to_idx[bs_tuple], param_idx] = pop
        
        self._draw_heatmap(ax, populations_heatmap, all_bitstrings, 
                          unique_params_arr, title_suffix)
    
    def _compute_populations(
            self,
            bitstrings: NDArray,
            all_bitstrings: list,
    ) -> tuple[dict, dict]:
        """Compute bitstring populations and counts from array of bitstrings.
        
        bitstrings: shape (n_shots, n_elements) with binary values
        Returns: (populations_dict, counts_dict)
        """
        # Convert each row to a tuple (bitstring)
        bitstring_tuples = [self._bitstring_to_tuple(bs) for bs in bitstrings]
        
        pops = {}
        counts = {}
        total = len(bitstring_tuples)
        
        for bs_tuple in set(bitstring_tuples):
            count = bitstring_tuples.count(bs_tuple)
            pops[bs_tuple] = count / total
            counts[bs_tuple] = count
        
        # Ensure all bitstrings represented
        for bs_tuple in all_bitstrings:
            if bs_tuple not in pops:
                pops[bs_tuple] = 0.0
                counts[bs_tuple] = 0
        
        return pops, counts
    
    def _get_populations_for_mask(
            self,
            shot_mask: np.ndarray,
            require_exact_rearrangement: bool,
            all_bitstrings: list,
    ) -> dict:
        """Compute bitstring populations for a given shot mask."""
        bitstrings = self.extract_bitstrings(
            shot_mask=shot_mask,
            require_exact_rearrangement=require_exact_rearrangement,
        )
        return self._compute_populations(bitstrings, all_bitstrings)
    
    def _draw_heatmap(
            self,
            ax: Axes,
            populations_heatmap: np.ndarray,
            all_bitstrings: list,
            unique_params_arr: np.ndarray,
            title_suffix: str,
    ):
        """Draw the heatmap visualization on given axes."""
        n_bitstrings = len(all_bitstrings)
        n_unique = unique_params_arr.shape[0]
        bitstring_labels = [self._tuple_to_bitstring(bs) for bs in all_bitstrings]
        
        im = ax.imshow(
            populations_heatmap,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest',
        )
        
        ax.set_yticks(np.arange(n_bitstrings))
        ax.set_yticklabels(bitstring_labels, fontsize=self.plot_config.label_font_size)
        
        ax.set_xticks(np.arange(n_unique))
        if n_unique <= 20:
            param_labels = [f'{param_val[0]:.3g}' for param_val in unique_params_arr]
        else:
            step = max(1, n_unique // 10)
            ax.set_xticks(np.arange(0, n_unique, step))
            param_labels = [f'{unique_params_arr[i, 0]:.3g}' for i in range(0, n_unique, step)]
        ax.set_xticklabels(param_labels, fontsize=self.plot_config.label_font_size, rotation=45, ha='right')
        
        ax.set_ylabel('Bitstring', fontsize=self.plot_config.label_font_size)
        ax.set_xlabel(f'{self.params[0].axis_label}', fontsize=self.plot_config.label_font_size)
        # ax.set_title(f'Bitstring Populations (Sites: {list(self.selected_sites)}){title_suffix}',
        #              fontsize=self.plot_config.title_font_size)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Population', fontsize=self.plot_config.label_font_size)
    
    def plot_bitstring_populations_curves(
            self,
            ax: Optional[Axes] = None,
            fig: Optional[Figure] = None,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
            average_shot_masks: bool = True,
            group_by_magnetization: bool = False,
            save_fig = True,
    ):
        """Plot bitstring populations vs parameter (one curve per bitstring).
        
        X-axis: unique parameter values, Y-axis: population, each bitstring is a curve.
        
        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on (ignored if multiple masks and average_shot_masks=False).
        fig : Figure, optional
            Figure to plot on (creates new if None and ax is None).
        shot_mask : np.ndarray or Sequence[np.ndarray], optional
            Additional mask(s) to filter shots. Multiple masks can be averaged
            or plotted separately.
        require_exact_rearrangement : bool, default=False
            Apply rearrangement mask based on target_sites at image 1.
        average_shot_masks : bool, default=True
            If True, average multiple masks. If False, create separate subplots.
        """
        # Validate masks if averaging
        if average_shot_masks:
            self._validate_shot_masks(shot_mask)
        
        has_multiple = (isinstance(shot_mask, (list, tuple)) and len(shot_mask) > 1) or \
                       (isinstance(shot_mask, np.ndarray) and shot_mask.ndim == 2 and shot_mask.shape[0] > 1)
        plot_separately = has_multiple and not average_shot_masks
        
        if plot_separately:
            # Create subplots for each mask
            n_masks = len(shot_mask)

            if fig is None:
                fig = plt.figure(
                    figsize=(self.plot_config.figure_size[0], 
                             self.plot_config.figure_size[1] * n_masks),
                    constrained_layout=self.plot_config.constrained_layout,
                )
            axes = fig.subplots(nrows=1, ncols=n_masks, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            
            # Handle both list/tuple and 2D array forms of multiple masks
            masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                         [shot_mask[i] for i in range(shot_mask.shape[0])]
            
            for mask_idx, mask in enumerate(masks_iter):
                self._plot_curves_single_mask(
                    ax=axes[mask_idx],
                    shot_mask=mask,
                    require_exact_rearrangement=require_exact_rearrangement,
                    group_by_magnetization=group_by_magnetization,
                    # title_suffix=f" (Mask {mask_idx+1})"
                )
        else:
            # Single mask, no mask, or averaging multiple masks
            if ax is None:
                if fig is None:
                    fig = plt.figure(
                        figsize=self.plot_config.figure_size,
                        constrained_layout=self.plot_config.constrained_layout,
                    )
                ax = fig.subplots()
            else:
                save_fig = False
            
            self._plot_curves_single_mask(
                ax=ax,
                shot_mask=shot_mask,
                require_exact_rearrangement=require_exact_rearrangement,
                group_by_magnetization=group_by_magnetization,
                # title_suffix=""
            )
            
        fig.suptitle(str(self.folder_path), fontsize=8)

        plt.tight_layout()
        
        if save_fig:
            figname = self.folder_path / 'bitstring_populations_curves.pdf'
            fig.savefig(figname)
            plt.show()
    
    def _plot_curves_single_mask(
            self,
            ax: Axes,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
            group_by_magnetization: bool = False,
            title_suffix: str = "",
    ):
        """Plot curves for one or more masks (averaged if multiple)."""
        # Determine if we're averaging multiple masks
        average_masks = (isinstance(shot_mask, (list, tuple))) or \
                        (isinstance(shot_mask, np.ndarray) and shot_mask.ndim == 2)
        
        # Get unique scan parameters
        unique_params_arr = self.unique_params()
        if unique_params_arr.ndim == 1:
            unique_params_arr = unique_params_arr[:, np.newaxis]
        
        n_unique = unique_params_arr.shape[0]
        
        # Determine bitstring size from the mask, not from selected_sites
        if average_masks:
            masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                         [shot_mask[i] for i in range(shot_mask.shape[0])]
            first_mask = masks_iter[0]
        else:
            first_mask = shot_mask
        
        # Get mask size
        if first_mask is None:
            n_sites = len(self.selected_sites)
        elif isinstance(first_mask, (list, np.ndarray)):
            n_sites = len(first_mask)
        else:
            # Boolean array - count True values
            n_sites = np.sum(first_mask)
        
        # Generate all possible bitstrings in lexicographic order
        from itertools import product
        all_bitstrings = [tuple(bs) for bs in product([0, 1], repeat=n_sites)]
        
        # Initialize data: dict[bitstring_tuple] = list of populations for each param
        populations_curves = {bs: np.zeros(n_unique) for bs in all_bitstrings}
        errors_curves = {bs: np.zeros(n_unique) for bs in all_bitstrings}
        
        # For each unique parameter value
        loop_params = self._loop_params()
        if loop_params.ndim == 1:
            loop_params = loop_params[:, np.newaxis]
        
        for param_idx, param_val in enumerate(unique_params_arr):
            matches_temp = np.all(loop_params == param_val[np.newaxis, :], axis=1)
            if require_exact_rearrangement:
                exact_mask = self._shot_mask_exact_rearrangement()
                matches = matches_temp[exact_mask]
            else:
                matches = matches_temp
            if not matches.any():
                continue
            
            if average_masks:
                # Average populations across multiple masks
                pops_list = []
                counts_list = []
                # Handle both list/tuple and 2D array forms of multiple masks
                masks_iter = shot_mask if isinstance(shot_mask, (list, tuple)) else \
                             [shot_mask[i] for i in range(shot_mask.shape[0])]
                for mask in masks_iter:
                    # Extract bitstrings with element mask, then filter by parameter
                    bitstrings = self.extract_bitstrings(
                        shot_mask=mask,
                        require_exact_rearrangement=require_exact_rearrangement,
                    )
                    # extract_bitstrings returns 2D array (shots, n_elements)
                    bitstrings_for_param = bitstrings[matches]
                    if len(bitstrings_for_param) > 0:
                        pops, counts = self._compute_populations(bitstrings_for_param, all_bitstrings)
                        pops_list.append(pops)
                        counts_list.append(counts)
                
                if pops_list:
                    for bs_tuple in all_bitstrings:
                        # Average populations across masks
                        avg_pop = np.mean([pops.get(bs_tuple, 0) for pops in pops_list])
                        populations_curves[bs_tuple][param_idx] = avg_pop
                        # Compute average error using binomial error formula
                        errors = []
                        for pops, counts in zip(pops_list, counts_list):
                            count = counts[bs_tuple]
                            total = sum(counts.values())
                            if total > 0:
                                # Laplace binomial error
                                laplace_p = (count + 1) / (total + 2)
                                error = np.sqrt(laplace_p * (1 - laplace_p) / (total + 2))
                                errors.append(error)
                        if errors:
                            errors_curves[bs_tuple][param_idx] = np.mean(errors)/np.sqrt(shot_mask.shape[0])# Very bad approx but good enough for now, MUST FIX!!!
            else:
                # Single mask or no mask
                bitstrings = self.extract_bitstrings(
                    shot_mask=shot_mask,
                    require_exact_rearrangement=require_exact_rearrangement,
                )
                # extract_bitstrings returns 2D array (shots, n_elements)
                bitstrings_for_param = bitstrings[matches]
                if len(bitstrings_for_param) > 0:
                    pops, counts = self._compute_populations(bitstrings_for_param, all_bitstrings)
                    total = len(bitstrings_for_param)
                    for bs_tuple, pop in pops.items():
                        populations_curves[bs_tuple][param_idx] = pop
                        # Compute binomial error
                        count = counts[bs_tuple]
                        if total > 0:
                            laplace_p = (count + 1) / (total + 2)
                            error = np.sqrt(laplace_p * (1 - laplace_p) / (total + 2))
                            errors_curves[bs_tuple][param_idx] = error
        
        # Plot curves with error bars
        param_values = unique_params_arr[:, 0]

        if group_by_magnetization:
            max_mag = np.size(all_bitstrings[0])+1
            colors = plt.cm.coolwarm(np.linspace(0, 1, max_mag))
            mag_curves = {mag: np.zeros(n_unique) for mag in range(max_mag)}
            mag_errors = {mag: np.zeros(n_unique) for mag in range(max_mag)}
            for mag in range(max_mag):
                for bs in all_bitstrings:
                    if sum(bs)==mag:
                        mag_curves[mag] += populations_curves[bs]
                        mag_errors[mag] += errors_curves[bs] / (comb(max_mag-1, mag)**(3/2))# Very bad approx but good enough for now, MUST FIX!!!

            for mag in range(max_mag-1, -1, -1):
                 ax.errorbar(param_values, mag_curves[mag], 
                       yerr=mag_errors[mag],
                       label=f"$S_z$ = {mag - (max_mag-1)/2}", color=colors[mag],
                       **self.plot_config.errorbar_kw)
            
        else:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(all_bitstrings)))
            for idx, bs_tuple in enumerate(all_bitstrings):
                bitstring_label = self._tuple_to_bitstring(bs_tuple)
                ax.errorbar(param_values, populations_curves[bs_tuple], 
                        yerr=errors_curves[bs_tuple],
                        label=bitstring_label, color=colors[idx],
                        **self.plot_config.errorbar_kw)
        
        ax.set_xlabel(f'{self.params[0].axis_label}', fontsize=self.plot_config.label_font_size)
        ax.set_ylabel('Population', fontsize=self.plot_config.label_font_size)
        # ax.set_title(f'Bitstring Populations (Sites: {list(self.selected_sites)}){title_suffix}',
        #              fontsize=self.plot_config.title_font_size)
        ax.legend(loc='upper right', fontsize=self.plot_config.label_font_size)
        ax.grid(True, alpha=0.3)

    def _fit__bitstring_curves(
            self,
            all_bitstrings: list,
            param_values: np.ndarray,
            bitstring_pops: dict[tuple, np.ndarray],
            bitstring_errors: Optional[dict[tuple, np.ndarray]] = None,
            fit_type: str = "GHZ Rabi"
    ):
        if fit_type == "GHZ Rabi":
            bs_0 = all_bitstrings[0]
            bs_1 = all_bitstrings[-1]

            pops_0 = bitstring_pops[bs_0]
            pops_1 = bitstring_pops[bs_1]
            pops_other = np.zeros(np.size(pops_0))
            for bs_tuple in all_bitstrings[1:-1]:
                pops_other += bitstring_pops[bs_tuple]

            if bitstring_errors != None:
                errs_0 = bitstring_errors[bs_0]
                errs_1 = bitstring_errors[bs_1]
                errs_other = np.zeros(np.size(pops_0))
                for bs_tuple in all_bitstrings[1:-1]:
                    errs_other += bitstring_errors[bs_tuple]

            
            def GHZ_oscillations(J, C, B, b, Gamma, t):
                pops_bad = B*t + b
                pop1 = 1/2*(1-np.exp(-Gamma*t)*np.cos(J*2*np.pi*t))
                pop0 = 1-pop1
                pop1 = pop1-pops_bad/2
                pop0 = pop0 - pops_bad/2

                return np.array([pop0, pop1, pops_bad]).flatten()
            
            pops_flat = np.array([pops_0, pops_1, pops_other]).flatten()

            # popt, pconv = 
            



        else:
            raise ValueError("Hey punk you know you can't fit that")
