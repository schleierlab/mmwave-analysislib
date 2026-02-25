from __future__ import annotations

import os
import re
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional, cast, overload
from typing_extensions import assert_never

import h5py  # type: ignore
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.typing as pdt
import uncertainties  # type: ignore
import uncertainties.unumpy as unp  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from analysislib.common.base_statistics import BaseStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig
from analysislib.common.typing import StrPath


class bidict(dict):
    """
    https://stackoverflow.com/a/21894086
    """
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


prefixes = bidict({
    'p': -12,
    'n': -9,
    'u': -6,
    'm': -3,
    '': 0,
    'k': +3,
    'M': +6,
    'G': +9,
})


def _decompose_unitstr(unitstr):
    units = ['Hz', 's']
    regex = f'({"|".join(prefixes.keys())})({"|".join(units)})'
    match = re.fullmatch(regex, unitstr)
    if match is None:
        raise ValueError(f'Unrecognized SI unit {repr(unitstr)}')
    prefix, baseunit = match.groups()

    return prefix, baseunit


def find_offset_and_scale(values, unitstr):
    maxval = np.max(np.abs(values))
    maxlog = np.log10(maxval)

    try:
        prefix, baseunit = _decompose_unitstr(unitstr)
    except ValueError:
        return 0, 1, unitstr

    exponent_offset = ((maxlog + 1) // 3) * 3
    new_exponent =  (prefixes[prefix] + exponent_offset)
    newprefix = prefixes.inverse[new_exponent][0]

    return 0, 10**exponent_offset, newprefix + baseunit


@dataclass(frozen=True)
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

    def axis_label(self, unit: Optional[str] = None):
        namestr = self.friendly_name if self.friendly_name is not None else self.name
        if unit is None:
            plot_unit = self.unit
        else:
            plot_unit = unit
        unitstr = f' ({plot_unit})' if plot_unit != '' else ''
        return f'{namestr}{unitstr}'


class ScanningParameters:
    params: tuple[ScanningParameter, ...]
    param_inds: dict[str, int]  # maybe just store dict[str, ScanningParameter]?

    def __init__(self, params: Sequence[ScanningParameter]):
        self.params = tuple(params)
        self.param_inds = {
            param.name: i
            for i, param in enumerate(self.params)
        }

    def __getitem__(self, key: int | str) -> ScanningParameter:
        if isinstance(key, int):
            index = key
            return self.params[index]
        elif isinstance(key, str):
            name = key
            return self.params[self.param_inds[name]]
        else:
            assert_never(key)

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    @classmethod
    def from_h5_tuples(cls, iterable: Iterable) -> ScanningParameters:
        return cls([ScanningParameter.from_h5_tuple(tup) for tup in iterable])


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

    n_runs: int
    '''Total number of shots in this expansion'''

    rearrangement: bool
    '''Whether data being analyzed uses tweezer rearrangement'''
    # TODO: rearrangement-related functionality should probably be handled by a subclass?

    site_occupancies: np.ndarray
    '''
    site_occupancies is of shape (num_shots, num_images, num_sites)
    '''

    target_sites: Sequence[int]

    # TODO clean this up
    KEY_SHOT: ClassVar[str] = 'shot'
    KEY_IMAGE: ClassVar[str] = 'image'
    KEY_SITE: ClassVar[str] = 'site'
    KEY_INITIAL: ClassVar[str] = 'initial'
    KEY_SURVIVAL: ClassVar[str] = 'survival'
    KEY_SURVIVAL_RATE: ClassVar[str] = 'survival_rate'
    KEY_SURVIVAL_RATE_STD: ClassVar[str] = 'survival_rate_std'

    def __init__(
            self,
            preproc_h5_path: StrPath,
            shot_h5_path: Optional[StrPath] = None,
            plot_config: Optional[PlotConfig] = None,
            *,
            rearrangement: bool = False,
            shot_index: int = -1,
            target_sites: Sequence[int] = [],
    ):
        super().__init__(preproc_h5_path=preproc_h5_path, shot_index=shot_index)
        self.target_sites = target_sites

        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(preproc_h5_path)
        # if shot_h5_path is not None:
        #     self._save_mloop_params(shot_h5_path)
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
            self.site_occupancies = np.asarray(f['site_occupancies'][:], dtype=bool)
            self.site_rois = ROI.fromarray(f['site_rois'])
            self.params_list = f['params'][:]
            self.n_runs = cast(int, f.attrs['n_runs'])
            self.current_params = f['current_params'][:]
            self.run_times_strs = np.char.decode(np.asarray(f['run_times'][:], dtype=bytes), encoding='utf-8')

            self.params = ScanningParameters.from_h5_tuples(self.params_list)

            do_rearrangement = f.attrs['do_rearrangement']
            if not (isinstance(do_rearrangement, bool) or isinstance(do_rearrangement, np.bool_)):
                raise TypeError
            self.rearrangement = do_rearrangement

    @property
    def shot_index(self) -> int:
        if self._shot_index == -1:
            return self.shots_processed - 1
        return self._shot_index

    @property
    def shots_processed(self) -> int:
        return self.site_occupancies.shape[0]

    @property
    def is_final_shot(self) -> bool:
        return self.shots_processed == self.n_runs

    @property
    def n_images(self) -> int:
        return self.site_occupancies.shape[1]

    @property
    def n_sites(self) -> int:
        return self.site_occupancies.shape[2]

    @property
    def initial_atoms_array(self):
        ind = 1 if self.rearrangement else 0
        return self.site_occupancies[:, ind, :]

    @property
    def surviving_atoms_array(self):
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        inds = slice(1, 3) if self.rearrangement else slice(0, 2)
        return self.site_occupancies[:, inds, :].prod(axis=-2)

    def dataframe(
        self,
        shot_mask: Optional[np.ndarray] = None,
        require_exact_rearrangement: bool = False,) -> pd.DataFrame:
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
        ----------
        shot_mask : optional boolean array of shape (shots,)
            If provided, only these shots are included.
        require_exact_rearrangement : bool
            When True (and self.rearrangement is True), only include shots where
            image index 1 matches the target pattern exactly.
        '''
        # figure out which shots to include
        if require_exact_rearrangement:
            exact_mask = self._shot_mask_exact_rearrangement()
            if shot_mask is None:
                shot_mask = exact_mask
            else:
                if shot_mask.shape != exact_mask.shape:
                    raise ValueError("shot_mask has wrong shape")
                shot_mask = shot_mask & exact_mask

        # Slice per-shot arrays and param table if a mask is active
        if shot_mask is not None:
            params_for_index = self.current_params[shot_mask]
            initial_arr = self.initial_atoms_array[shot_mask]
            survival_arr = self.surviving_atoms_array[shot_mask]
        else:
            params_for_index = self.current_params
            initial_arr = self.initial_atoms_array
            survival_arr = self.surviving_atoms_array

        # --- original dataframe construction ---
        index = pd.MultiIndex.from_arrays(
            params_for_index.T,
            names=[param.name for param in self.params],
        )

        def assemble_occupancy_df(array: NDArray, name: str):
            df = pd.DataFrame(array, index=index, dtype=bool)
            df.columns.name = self.KEY_SITE
            occupancy_df = df.stack()
            occupancy_df.name = name
            return occupancy_df

        df_initial = assemble_occupancy_df(initial_arr, name=self.KEY_INITIAL)
        df_survival = assemble_occupancy_df(survival_arr, name=self.KEY_SURVIVAL)
        df = pd.concat([df_initial, df_survival], axis=1)

        # index = pd.MultiIndex.from_arrays(
        #     self.current_params.T,
        #     names=[param.name for param in self.params],
        # )

        # def assemble_occupancy_df(array: NDArray, name: str):
        #     df = pd.DataFrame(array, index=index, dtype=bool)
        #     df.columns.name = self.KEY_SITE
        #     occupancy_df = df.stack()
        #     occupancy_df.name = name
        #     return occupancy_df

        # df_initial = assemble_occupancy_df(self.initial_atoms_array, name=self.KEY_INITIAL)
        # df_survival = assemble_occupancy_df(self.surviving_atoms_array, name=self.KEY_SURVIVAL)
        # df = pd.concat([df_initial, df_survival], axis=1)

        return df.reset_index()

    def run_time_series(self) -> pd.Series:
        '''
        Return the run timestamps of the shots in this analysis.

        Returns
        -------
        pandas.Series
            A series of all the timestamps, as datetime64 objects.
        '''
        return pd.Series(self.run_times_strs, dtype='datetime64[ns]')

    # this is intended to supersede the above dataframe() eventually, since it has shot number information
    def series(self) -> pd.Series:
        # ignoring typechecker pending pandas-stubs#1285
        mi = pd.MultiIndex.from_product(
            [range(self.shots_processed), range(self.n_images), range(self.n_sites)],  # type: ignore
            sortorder=0,
            names=[self.KEY_SHOT, self.KEY_IMAGE, self.KEY_SITE],
        )
        return pd.Series(data=self.site_occupancies.flatten(), index=mi, name='occupancy', dtype=bool)

    def scan_param_df(self) -> pd.DataFrame:
        index = pd.RangeIndex(self.shots_processed, name=self.KEY_SHOT)
        return pd.DataFrame(
            self.current_params,
            index=index,
            columns=[param.name for param in self.params],
        )

    def dataframe_binomial_error(
            self,
            data,
            success_key: str,
            total_key: str,
            method: Literal['laplace'] = 'laplace',
            name: str = 'binomial_error') -> None:
        '''
        Given a DataFrame-like object with columns summarizing the outcomes of Bernoulli trials
        (that is, total samples "n" and successes "a"), produce a new column giving an uncertainty estimate
        on the success rate.
        '''
        success = data[success_key]
        total = data[total_key]

        if method == 'laplace':
            laplace_p = (success + 1) / (total + 2)
            new_col = np.sqrt(laplace_p * (1 - laplace_p) / (total + 2))

        data[name] = new_col

    @overload
    def dataframe_survival(self, data: pd.DataFrame) -> pd.Series: ...
    @overload
    def dataframe_survival(self, data: pdt.DataFrameGroupBy) -> pd.DataFrame: ...

    def dataframe_survival(self, data):
        df = data[[self.KEY_INITIAL, self.KEY_SURVIVAL]].sum()

        surv = df[self.KEY_SURVIVAL]
        total = df[self.KEY_INITIAL]
        df[self.KEY_SURVIVAL_RATE] = surv / total

        self.dataframe_binomial_error(df, self.KEY_SURVIVAL, self.KEY_INITIAL, name=self.KEY_SURVIVAL_RATE_STD)
        return df

    def _shot_mask_exact_rearrangement(self) -> np.ndarray:
        """
        True for shots where image 1 matches target_sites exactly.
        If not rearrangement mode or no target sites set, returns all True.
        """
        if (not self.rearrangement) or (len(self.target_sites) == 0):
            return np.ones(self.shots_processed, dtype=bool)

        target_bool = np.zeros(self.n_sites, dtype=bool)
        target_bool[np.asarray(self.target_sites, dtype=int)] = True

        img1 = self.site_occupancies[:, 1, :]  # (shots, sites)
        return np.all(img1 == target_bool[None, :], axis=1)
    # LEGACY CODE

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

    def plot_rearrange_histagram(self, target_array, ax: Axes, plot_overlapping_histograms: bool = True):
        '''
        Plots a histogram of the number of sites in the taerget array after rearrangement.

        Parameters
        ----------
        target_array : array_like
            Array of target sites.
        ax : matplotlib.axes.Axes,
            Axes object to plot on.
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

        zero_atom_in_target_indices = np.where(atom_count_in_target_list[1] == 0)[0]
        print('n_shots (total experiment)', n_shots)
        print('n_rarrange_shots', n_rearrange_shots)
        print('success_rearrange', success_rearrange)
        print(f"Rearrange attempts with 0 loaded atoms: {zero_atom_in_target_indices.size}")
        print(f"Indices: {zero_atom_in_target_indices}")


    def plot_rearrange_site_success_rate(self, target_array, ax: Axes):
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

    def plot_site_loading_rates(self, ax: Axes):
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

    def _pairs_from_targets(
        self,
        drop_last_incomplete: bool = True,
        explicit_pairs: Optional[Sequence[tuple[int, int]]] = None):
        pair_size = 2
        if explicit_pairs is not None:
            pairs = np.asarray(explicit_pairs, dtype=int)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("explicit_pairs must be a list/tuple of (i,j) pairs.")
            return pairs

        tgt = list(self.target_sites)  # preserve given order
        if len(tgt) < pair_size:
            return np.empty((0, 2), dtype=int)

        remainder = len(tgt) % pair_size
        if remainder:
            if drop_last_incomplete:
                tgt = tgt[:len(tgt) - remainder]
            else:
                raise ValueError(
                    f"target_sites length ({len(self.target_sites)}) not divisible by pair_size={pair_size}"
                )

        chunks = [tuple(tgt[i:i+pair_size]) for i in range(0, len(tgt), pair_size)]
        pairs: list[tuple[int, int]] = []
        for ch in chunks:
            if len(ch) != 2:
                raise ValueError("pair_size must be 2 for pair-state populations.")
            pairs.append((int(ch[0]), int(ch[1])))
        return np.asarray(pairs, dtype=int)

    def _plot_pair_state_population(
        self,
        ax,
        indep_var,
        require_exact_rearrangement: bool = True,
        explicit_pairs: Optional[Sequence[tuple[int, int]]] = None):
        if not self.rearrangement:
                raise ValueError("plot_pair_states=True requires rearrangement=True and 3 images (0,1,2).")
        if self.n_images < 3:
            raise ValueError("Expected 3 images (indices 0,1,2) for pair-state populations.")
        if (len(self.target_sites) < 2) and (explicit_pairs is None):
            raise ValueError("Need at least two target sites or provide explicit_pairs for pair-state plot.")

        # Use the same shot mask logic used in dataframe()
        if require_exact_rearrangement:
            mask = self._shot_mask_exact_rearrangement()
            if not mask.any():
                ax.text(0.5, 0.5, "No exact-matching rearranged shots", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_axis_off()
        else:
            mask = np.ones(self.shots_processed, dtype=bool)

        # Build pairs in GIVEN ORDER (or explicit)
        pairs = self._pairs_from_targets(explicit_pairs=explicit_pairs)
        if pairs.size == 0:
            raise ValueError("No complete pairs could be formed for pair-state plot.")
        n_pairs = pairs.shape[0]

        # image-2 occupancy (S-survival) and the scan parameter (x-values) for those shots
        img2 = self.site_occupancies[mask, 2, :] # (shots_kept, n_sites)
        xvals = self.current_params[mask, 0] # (shots_kept,)

        # per pair per shot
        SS_list, PP_list, PS_list, SP_list = [], [], [], []
        for a_idx, b_idx in pairs:
            a2 = img2[:, a_idx]
            b2 = img2[:, b_idx]
            SS_list.append(a2 & b2)
            PP_list.append((~a2) & (~b2))
            PS_list.append((~a2) & b2)
            SP_list.append(a2 & (~b2))

        SS = np.vstack(SS_list)  # (n_pairs, shots_kept)
        PP = np.vstack(PP_list)
        PS = np.vstack(PS_list)
        SP = np.vstack(SP_list)

        x_arr = np.asarray(indep_var)
        k_SS = np.zeros_like(x_arr, dtype=float)
        k_PP = np.zeros_like(x_arr, dtype=float)
        k_PS = np.zeros_like(x_arr, dtype=float)
        k_SP = np.zeros_like(x_arr, dtype=float)
        N = np.zeros_like(x_arr, dtype=float)

        for i, xv in enumerate(x_arr):
            bin_idx = (xvals == xv) # which kept shots belong to this x
            n_shots_bin = int(bin_idx.sum())
            N[i] = n_pairs * n_shots_bin
            if N[i] == 0:
                continue
            k_SS[i] = SS[:, bin_idx].sum() # (n_pairs, n_shots_bin)
            k_PP[i] = PP[:, bin_idx].sum()
            k_PS[i] = PS[:, bin_idx].sum()
            k_SP[i] = SP[:, bin_idx].sum()

        def division(a, b):
            # aviod devision by 0
            # return a/b if b > 0, else return nan
            return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=(b > 0))

        def prop_and_std(k, n):
            p = division(k, n)
            lap = division(k + 1, n + 2)
            std = np.sqrt(lap * (1 - lap) / (n + 2,))
            return p, std

        pSS, eSS = prop_and_std(k_SS, N)
        pPP, ePP = prop_and_std(k_PP, N)
        pPS, ePS = prop_and_std(k_PS, N)
        pSP, eSP = prop_and_std(k_SP, N)

        pD = pPP - pSS # differences
        pEO = pPP + pSS - pSP - pPS  # even-odd

        var_D = division(pPP + pSS - pD**2, N)
        var_EO = division(1-pEO**2, N)

        eD  = np.sqrt(var_D)
        eEO = np.sqrt(var_EO)

        ax.errorbar(x_arr, pSS, yerr=eSS, label="SS", **self.plot_config.errorbar_kw)
        ax.errorbar(x_arr, pPS, yerr=ePS, label="PS", **self.plot_config.errorbar_kw)
        ax.errorbar(x_arr, pSP, yerr=eSP, label="SP", **self.plot_config.errorbar_kw)
        ax.errorbar(x_arr, pPP, yerr=ePP, label="PP", **self.plot_config.errorbar_kw)
        # ax.errorbar(x_arr, pD, yerr = eD, label = 'PP-SS', **self.plot_config.errorbar_kw)
        # ax.errorbar(x_arr, pEO, yerr = eEO, label = 'Even-Odd', **self.plot_config.errorbar_kw)

        ax.set_xlabel(self.params[0].axis_label, fontsize=self.plot_config.label_font_size)
        ax.set_ylabel("Pair population", fontsize=self.plot_config.label_font_size)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.tick_params(axis="both", which="major", labelsize=self.plot_config.label_font_size)

        # file_path = os.path.join(f"{self.folder_path}/", '2025-10-01-0004_ramsey_dimer_data.npz')
        # np.savez(file_path,
        #  x_arr=x_arr,
        #  pSS=pSS, eSS=eSS,
        #  pPP=pPP, ePP=ePP,
        #  pPS=pPS, ePS=ePS,
        #  pSP=pSP, eSP=eSP,
        #  pD=pD, eD=eD,
        #  pEO=pEO, eEO=eEO, N=N)

    def _save_mloop_params(self, shot_h5_path: str | os.PathLike) -> None:
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
        run = lyse.Run(h5_path=shot_h5_path)  # type: ignore
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

    # REFACTORED

    def plot_survival_rate_1d(
        self,
        fig: Figure,
        fit_type = None,
        require_exact_rearrangement: bool = False,
        plot_pair_states: bool = False,
        explicit_pairs: Optional[Sequence[tuple[int, int]]] = None,
        show_hist: bool = False,
        averaging_window: Optional[int] = None,
    ):
        """
        Plot survival rate over all sites (1D scan).
        Optionally add a second subplot below that shows pair-state populations
        (SS /PP /PS /SP ) computed from image index 2.

        fig: Figure
            A matplotlib Figure to plot in.
        plot_lorentz: bool
            If True and we're analyzing the last shot,
            also fit a Lorentzian lineshape, plot it,
            and show the fit parameters.
        show_hist: bool
            If True, show to the right of the main plot
            a histogram of all the values encountered.
        averaging_window: int, optional
            If provided, also plot the average of the `averaging_window` most recent.
            
        Returns:
            indep_var, survival_rates, survival_rate_errs
        """
        if show_hist:
            _ax_plot, _ax_hist = fig.subplots(
                ncols=2,
                nrows=1,
                sharey=True,
                gridspec_kw=dict(width_ratios=[3,1]),
            )
            ax_plot = cast(Axes, _ax_plot)
            ax_hist = cast(Axes, _ax_hist)
        elif plot_pair_states:
            _ax_plot, _ax_pairs = fig.subplots(
                ncols=2,
                nrows=1,
                sharey=True,
            )
            ax_plot = cast(Axes, _ax_plot)
            ax_pairs = cast(Axes, _ax_pairs)
            self.plot_config.configure_grids(ax_pairs)
            ax_pairs.set_ylim(bottom=0)
        else:
            ax_plot = fig.subplots()

        ax_plot.set_ylim(bottom=0)
        self.plot_config.configure_grids(ax_plot)

        fig.suptitle(str(self.folder_path), fontsize=8)

        # build dataframe (optionally filtered to exact rearrangement at image 1)
        df = self.dataframe(require_exact_rearrangement=require_exact_rearrangement)

        if len(self.params) != 1:
            raise ValueError("plot_survival_rate_1d expects exactly one scanned parameter")
        elif len(self.params) == 1:
            gb = df.groupby([param.name for param in self.params])
            unitstr = self.params[0].unit
        elif len(self.params) == 0:
            gb = df
            unitstr = ''
        survival_df = self.dataframe_survival(gb)

        indep_var = survival_df.index
        offset, xscale, scaled_unit = find_offset_and_scale(indep_var, unitstr)

        if len(self.params) == 1:
            xlabel = self.params[0].axis_label(unit=scaled_unit)
        elif len(self.params) == 0:
            xlabel = 'Shot number'
        ax_plot.set_xlabel(xlabel, fontsize=self.plot_config.label_font_size)

        indep_var_scaled = indep_var / xscale
        survival_rates = survival_df[self.KEY_SURVIVAL_RATE]
        survival_rate_errs = survival_df[self.KEY_SURVIVAL_RATE_STD]
        ax_plot.errorbar(
            indep_var_scaled,
            survival_rates,
            yerr=survival_rate_errs,
            **self.plot_config.errorbar_kw,
        )

        ax_plot.set_ylabel(
            'Survival rate',
            fontsize=self.plot_config.label_font_size,
        )
        ax_plot.tick_params(
            axis='both',
            which='major',
            labelsize=self.plot_config.label_font_size,
        )

        ax_plot.set_title('Survival rate over all sites', fontsize=self.plot_config.title_font_size)

        if show_hist:
            ax_hist.hist(
                survival_rates,
                orientation='horizontal',
            )
        if averaging_window is not None:
            # TODO add errorbars to this
            ax_plot.plot(
                indep_var_scaled,
                survival_rates.rolling(averaging_window).mean(),
                color='C1',
                marker='.',
            )

        if self.is_final_shot and fit_type is not None and len(indep_var) > 2:
            x_plot = np.linspace(np.min(indep_var), np.max(indep_var), 1000)
            x_plot_scaled = x_plot / xscale

            if fit_type == 'lorentzian':
                popt, pcov = self.fit_lorentzian(indep_var, survival_rates, sigma=survival_rate_errs, peak_direction=-1)
                upopt = uncertainties.correlated_values(popt, pcov)
                ax_plot.plot(x_plot_scaled, self.lorentzian(x_plot, *popt))
                fig.suptitle(
                    f'Center frequency: ${upopt[0]:SL}$ {self.params[0].unit}; Width: ${upopt[1]:SL}$ {self.params[0].unit}, amplitude: ${4 * upopt[2] / upopt[1]:SL}$'
                )
            elif fit_type == 'quadratic':
                popt, pcov = self.fit_quadratic(indep_var, survival_rates, sigma=survival_rate_errs, peak_direction=+1)
                upopt = uncertainties.correlated_values(popt, pcov)
                ax_plot.plot(x_plot_scaled, self.quadratic(x_plot, *popt), color = 'r')
                fig.suptitle(
                    f'Center: ${upopt[2]:SL}$ {self.params[0].unit}; offset: ${upopt[1]:SL}$'
                )
            elif fit_type == 'rabispec':
                popt, pcov = self.fit_rabispec(indep_var, survival_rates, sigma=survival_rate_errs, peak_direction=-1)
                upopt = uncertainties.correlated_values(popt, pcov)

                freq_unit = self.params[0].unit
                label = textwrap.dedent(f'''\
                    transition at ${upopt[0]:SL}$ {freq_unit}
                    $\Omega/2\pi = {upopt[1]:SL}$ {freq_unit}
                    effective pulse length ${upopt[2]:SL}$ ({freq_unit})$^{{-1}}$
                    contrast ${upopt[3]:SL}$, offset ${upopt[4]:SL}$'''
                )
                ax_plot.plot(x_plot_scaled, self.rabi_spectrum_model(x_plot, *popt), color='r', label=label)
                ax_plot.legend(fontsize='x-small')

        if self.is_final_shot and self.params[0].name == 'repetition_index':
            mean_df = self.dataframe_survival(df)
            umean = uncertainties.ufloat(mean_df[self.KEY_SURVIVAL_RATE], mean_df[self.KEY_SURVIVAL_RATE_STD])
            ax_plot.axhline(umean.n, label=f'Mean: {umean:S}')
            ax_plot.legend()

        if plot_pair_states:
            self._plot_pair_state_population(ax_pairs, indep_var_scaled, require_exact_rearrangement, explicit_pairs)

        return indep_var, survival_rates, survival_rate_errs

    def plot_survival_rate_2d(
            self,
            fig: Figure,
            plot_gaussian: bool = False,
    ):
        '''
        For higher dimensional scans,
        we will only plot the 2D cut corresponding to the two innermost scan variables.
        '''
        (ax1,), (ax2,) = fig.subplots(nrows=2, ncols=1, sharex=True, sharey=True, squeeze=False)

        # unique_params = self.unique_params()
        # x_params_index, y_params_index = self.get_params_order(unique_params)

        df = self.dataframe()
        groupby = df.groupby([param.name for param in self.params])
        survival_df = self.dataframe_survival(groupby)

        def df_to_pcolor_args(df: pd.Series | pd.DataFrame):
            # TODO consider moving sorting logic to self.dataframe
            this_run_params = self.current_params[-1]
            param_order = self.get_params_order(self.unique_params())[::-1]
            index_order = [self.params[ind].name for ind in param_order]
            df_reordered = df.reorder_levels(index_order).sort_index()

            cross_section = tuple(this_run_params[ind] for ind in param_order[:-2])
            twodim_df = df_reordered.xs(key=cross_section)

            unstack = twodim_df.unstack()
            if not isinstance(unstack, pd.DataFrame):
                raise ValueError('df must have a MultiIndex')
            
            pcolor_args = (
                unstack.columns,
                unstack.index,
                unstack,
            )
            cross_section_info = {
                self.params[ind]: this_run_params[ind]
                for ind in param_order[:-2]
            }

            return pcolor_args, cross_section_info

        def plot_key_2d(df: pd.Series | pd.DataFrame, ax: Axes):
            '''
            Produce a 2D plot of the data in a pandas dataframe,
            provided in long format with a MultiIndex for the two independent variables.

            Parameters
            ----------
            df: pd.DataFrame or pd.Series
                DataFrame with one column (or Series) and a 2D MultiIndex
            ax: Axes
                Axes in which to plot the data.
            '''
            (cols, ind, data), cross_section_info = df_to_pcolor_args(df)

            subplotspec = ax.get_subplotspec()
            if subplotspec is None:
                raise ValueError
            if subplotspec.is_last_row():
                ax.set_xlabel(
                    self.params[cols.name].axis_label(),
                    fontsize=self.plot_config.label_font_size,
                )
            if subplotspec.is_first_col():
                ax.set_ylabel(
                    self.params[ind.name].axis_label(),
                    fontsize=self.plot_config.label_font_size,
                )

            if subplotspec.is_first_row() and len(cross_section_info) > 0:
                cross_section_str = '; '.join(
                    f'{param.name} = {varval} {param.unit}'
                    for param, varval in cross_section_info.items()
                )
                ax.set_title(f'(at {cross_section_str})')
            return ax.pcolormesh(cols, ind, data, shading='nearest')

        pcolor_survival_rate = plot_key_2d(survival_df[self.KEY_SURVIVAL_RATE], ax1)
        pcolor_std = plot_key_2d(survival_df[self.KEY_SURVIVAL_RATE_STD], ax2)

        surv_cb = fig.colorbar(pcolor_survival_rate, ax=ax1)
        surv_cb.set_label('Survival rate')
        std_cb = fig.colorbar(pcolor_std, ax=ax2)
        std_cb.set_label('Uncertainty')

        for ax in [ax1, ax2]:
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.plot_config.label_font_size,
            )
            self.plot_config.configure_grids(ax)
        if plot_gaussian:
            popt, pcov = self.fit_gaussian_2d(*df_to_pcolor_args(survival_df[self.KEY_SURVIVAL_RATE]))
            perr = np.sqrt(np.diag(pcov))
            ax1.title.set_text(f'X waist = {popt[3]:.2f} +/- {perr[3]:.2f}, Y waist = {popt[4]:.2f} +/- {perr[4]:.2f}')

    def plot_survival_rate(
            self,
            fig: Optional[Figure] = None,
            fit_type_1d = None,
            plot_gaussian: bool = False,
            require_exact_rearrangement: bool = False,
            plot_pair_states: bool = False,
            show_hist: bool = False,
    ):
        """
        Plots the total survival rate of atoms in the tweezers, summed over all sites.

        Parameters
        ----------
        fig : Optional[Figure]
            The figure to plot on. If None, a new figure is created.
        """
        loop_params = self._loop_params()

        is_subfig = (fig is not None)
        if fig is None:
            fig = plt.figure(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )

        if loop_params.ndim == 0:
            self.plot_survival_rate_1d(
                fig,
                require_exact_rearrangement=require_exact_rearrangement,
                plot_pair_states=plot_pair_states,
                show_hist=True,
                averaging_window=10,
            )
        if loop_params.ndim == 1:
            self.plot_survival_rate_1d(
                fig,
                fit_type=fit_type_1d,
                require_exact_rearrangement=require_exact_rearrangement,
                plot_pair_states=plot_pair_states,
                show_hist=show_hist,
                averaging_window=None,
            )
        elif loop_params.ndim == 2:
            self.plot_survival_rate_2d(fig, plot_gaussian)
        else:
            raise NotImplementedError("I only know how to plot 1d and 2d scans")

        figname = self.folder_path / 'survival_rate_vs_param.pdf'
        if not is_subfig:
            fig.savefig(figname)


    # TODO this should be refactored with the other pandas-based
    # binomial error uncertainty implementations...
    # need to figure out best way to do this
    @staticmethod
    def binomial_rate_uncert(arr):
        trues = arr.sum()
        total = len(arr)
        mean = trues / total

        laplace_p = (trues + 1) / (total + 2)
        uncert = np.sqrt(laplace_p * (1 - laplace_p) / total)
        return uncertainties.ufloat(mean, uncert)

    def plot_loading_rate(self, ax: Axes):
        series = self.series()
        gb = series.xs(0, level=self.KEY_IMAGE).groupby(self.KEY_SHOT)

        agg = gb.agg(self.binomial_rate_uncert)

        run_time_series = self.run_time_series()
        '''index: shot number; values: run times'''

        ax.errorbar(
            run_time_series,
            unp.nominal_values(agg),
            yerr=unp.std_devs(agg),
            **self.plot_config.errorbar_kw,
        )

        ax.set_ylabel('Loading rate')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='dashed')
        ax.axhline(
            np.average(unp.nominal_values(agg)),
            color='0.5',
            label=f'Mean: {np.average(agg):S}',
        )
        ax.legend()
    
    def plot_loading_rate_1d(self, ax: Axes):
        df = self.dataframe()
        gb = df.groupby([param.name for param in self.params])[self.KEY_INITIAL]
        xlabel = self.params[0].axis_label()
        loading_rates_unc = gb.agg(self.binomial_rate_uncert)

        ax.errorbar(
            loading_rates_unc.index,
            unp.nominal_values(loading_rates_unc),
            yerr=unp.std_devs(loading_rates_unc),
            **self.plot_config.errorbar_kw,
        )

        ax.set_ylabel('Loading rate')
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='dashed')
        ax.legend()

    def _setup_shot_index_secax(self, ax: Axes):
        run_time_nums = mdates.date2num(self.run_time_series())
        run_time_nums_interp = np.empty((self.shots_processed + 2,))
        run_time_nums_interp[1:-1] = run_time_nums
        time_range = max(1, run_time_nums[-1] - run_time_nums[0])  # in case run_time_nums is len 1
        run_time_nums_interp[0] = run_time_nums[0] - time_range
        run_time_nums_interp[-1] = run_time_nums[-1] + time_range
        # [A, time0, time1, time2, time(n-1), B]
        # where A, time0, time(n-1), B are in arithmetic sequence

        shot_indices_interp = np.arange(-1, self.shots_processed + 1, dtype=np.float64)
        eps = 0.1
        shot_indices_interp[0] = -eps
        shot_indices_interp[-1] = shot_indices_interp[-2] + eps
        # [-0.1, 0, 1, 2, ..., n-1, n-1 + 0.1]
        # endpoints are so things at the far end don't look wrong

        def time_to_index(datenum):
            retval = np.interp(datenum, run_time_nums_interp, shot_indices_interp)
            return retval

        def index_to_time(index):
            time_num = np.interp(index, shot_indices_interp, run_time_nums_interp)
            return time_num

        secax = ax.secondary_xaxis(location='top', functions=(time_to_index, index_to_time))
        secax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10], integer=True))
        secax.set_xlabel('Shot number')

    def plot_rearrangement_performance(self, ax: Axes):
        if not self.rearrangement:
            raise ValueError

        REARRANGED_IMG_INDEX = 1
        occupancies_rearr = self.series().xs(REARRANGED_IMG_INDEX, level='image')
        target_sites_filter = occupancies_rearr.index.isin(self.target_sites, level=self.KEY_SITE)
        target_loading_rates = occupancies_rearr[target_sites_filter].groupby(self.KEY_SHOT).mean()

        ax.plot(
            self.run_time_series(),
            target_loading_rates,
            marker='.',
            linestyle='',
        )
        ax.set_ylabel('Array preparation fidelity')

    def plot_tweezing_statistics(self, fig: Optional[Figure] = None, avg_loading_rate: bool = False):
        rows = 2 if self.rearrangement else 1

        if fig is None:
            fig, axs = plt.subplots(nrows=rows, ncols=1, sharex=True)
        else:
            axs = fig.subplots(nrows=rows, ncols=1, sharex=True)
        axs = np.atleast_1d(axs)

        fig.suptitle('Tweezing statistics')
        if avg_loading_rate:
            self.plot_loading_rate_1d(axs[0])
        else:
            self.plot_loading_rate(axs[0])

            if self.rearrangement:
                self.plot_rearrangement_performance(axs[1])

            self._setup_shot_index_secax(axs[0])
            last_ax = axs[-1]
            # https://stackoverflow.com/a/56139690
            last_ax.set_xticks(last_ax.get_xticks(), last_ax.get_xticklabels(), rotation=45, ha='right')  # type: ignore
            last_ax.set_xlabel('Shot time')

        fig.align_ylabels()

    def counts_scatterplot(self, ax: Axes):
        ax.scatter(
            self.camera_counts[:, 0, :].flatten(),
            self.camera_counts[:, 1, :].flatten(),
            s=1,
        )
        ax.set_aspect(1)
        ax.set_xlabel('Image 0 counts')
        ax.set_ylabel('Image 1 counts')

    # LEGACY CODE

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

    def loop_param_and_site_survival_rate_matrix(self, num_time_groups = 1):
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
                # print('s_sum: ', s_sum)
                # print('i_sum: ', i_sum)
                zero_sites = np.where(i_sum == 0)[0]
                if zero_sites.size > 0:
                    print(f"Warning: No initial atoms detected at sites {zero_sites} for param {x}, group {g}")
                with np.errstate(divide='ignore', invalid='ignore'):
                    rate = np.true_divide(s_sum, i_sum)
                    rate[~np.isfinite(rate)] = np.nan
                    survival_rates[:, i, g] = rate

                # Laplace method
                # One shouldn't use laplace method here, because the sums are still small for each site
                # survival_rates[:, i, g] = (s_sum + 1) / (i_sum + 2)

        return unique_params, survival_rates

    # TODO: merge this into plot_survival_rate_by_site
    def plot_survival_rate_by_site_2d(
            self,
            ax: Optional[Axes] = None,
            plot_grouped_averaged: bool = False,
    ): #TODO: add grouped averaged option
        """
        Plots the survival rate of atoms in the tweezers, site by site.

        Parameters
        ----------
        ax: Axes, optional
            Axes to plot on. If not supplied, a new figure and axes are created.
        """
        is_subfig = (ax is not None)
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self.plot_config.figure_size,
                constrained_layout=self.plot_config.constrained_layout,
            )

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

        ax.set_xlabel(self.params[0].axis_label())
        ax.set_ylabel('Site index')
        fig.colorbar(pm, ax=ax)

        if not is_subfig:
            fig.savefig(f"{self.folder_path}/survival_rate_by_site_2d.pdf")
            fig.suptitle(f"{self.folder_path}")

    def group_data(self, data, group_size):
        n_groups = data.shape[0]//group_size
        print('n_groups',n_groups)
        grouped_data = data[:data.shape[0]].reshape(n_groups, group_size, -1)
        # averaged_data = grouped_data.mean(axis = 1)
        averaged_data =np.nanmean(grouped_data, axis = 1)

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
            if group_size == data.shape[0]:
                 ax = axs
            else:
                ax = axs[-i-1]
            ax.plot(unique_params, averaged_data[i],'.-',label = rf'group {i} data')
            path = os.path.join(f"{self.folder_path}/", f"data_multishot_{i}.csv")
            np.savetxt(path, [unique_params, averaged_data[i]], delimiter=",")
            if fit_type == 'rabi_oscillation':
                # Fit the model to the data
                initial_guess = [1, 2*np.pi*2e6, 0, 6e-6, 0.5]
                popt, pcov = curve_fit(self.rabi_model, unique_params, averaged_data[i], p0=initial_guess)

                # Extract fit results
                A_fit, Omega_fit, phi_fit, T2_fit, C_fit = popt

                upopt = uncertainties.correlated_values(popt, pcov)

                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1000,
                )
                ax.plot(x_plot, self.rabi_model(x_plot, *popt), 'r-', label=rf'{i}Fit')
                annotation_text = (
                    f'p-p Ampl: {A_fit*2:.3f}\n'
                    Rf'$\Omega$: $2\pi\times {upopt[1] / 1e6 / (2 * np.pi):SL}\,\mathrm{{MHz}}$'
                    '\n'
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
            elif fit_type == 'exponential': # fit lifetime
                initial_guess = [0.5,0.2,0]
                popt, pcov = curve_fit(self.exponential, unique_params, averaged_data[i], p0=initial_guess)
                upopt = uncertainties.correlated_values(popt, pcov)
                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1000,
                )

                ax.plot(x_plot, self.exponential(x_plot, *popt), 'r-', label=rf'{i}Fit')
                annotation_text = (
                    f'Lifetime: ${upopt[1]:SL}$ s\n'
                    f'Factor: ${upopt[0]:SL}$\n'
                    f'Offset: ${upopt[2]:SL}$\n' 
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
            elif fit_type == 'quadratic':
                # Fit the model to the data
                y_range = np.max(averaged_data[i]) - np.min(averaged_data[i])
                x_range = np.max(unique_params) - np.min(unique_params)
                guess = (y_range / x_range**2, np.min(averaged_data[i]), np.mean(unique_params))
                params_opt, params_cov = curve_fit(self.quadratic, unique_params, averaged_data[i], p0 = guess)

                # Extract fit results
                A_fit, B_fit, x_0_fit = params_opt
                print(A_fit, B_fit, x_0_fit)
                upopt = uncertainties.correlated_values(params_opt, params_cov)


                x_plot = np.linspace(
                    np.min(unique_params),
                    np.max(unique_params),
                    1001,
                )
                ax.plot(x_plot, self.quadratic(x_plot, *params_opt), 'r-', label=rf'{i}Fit')
                annotation_text = (
                    f'Center: {upopt[2]}'
                )

                ax.annotate(
                    annotation_text,
                    xy=(0.02, 0.05),  # Changed to bottom-left corner (x=0.02, y=0.05)
                    xycoords='axes fraction',
                    fontsize=9,
                    ha='left',
                    va='bottom',
                )

                ax.legend(loc='upper right')
        fig.supxlabel(self.params[0].axis_label())
        fig.supylabel('Population')

        # fig.suptitle("Rabi Oscillation Fits", fontsize=14)

        fig.savefig(f"{self.folder_path}/grouped_survival_rate_by_site_1d.pdf")
        fig.suptitle(f"{self.folder_path}")
