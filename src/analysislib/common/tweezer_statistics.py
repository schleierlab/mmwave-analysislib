from __future__ import annotations

import logging
import os
import re
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union, cast, overload

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing_extensions import assert_never

from analysislib.common.base_statistics import BaseStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig
from analysislib.common.typing import StrPath

logger = logging.getLogger(__name__)

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
    noop_retval = (0, 1, unitstr)
    if len(values) == 0:
        return noop_retval

    maxval = np.max(np.abs(values))
    if maxval == 0:
        return noop_retval

    maxlog = np.log10(maxval)

    try:
        prefix, baseunit = _decompose_unitstr(unitstr)
    except ValueError:
        return noop_retval

    if baseunit in ['Hz']:
        min_cutoff = 1
    elif baseunit in ['s']:
        min_cutoff = 0.1

    exponent_offset = ((maxlog - np.log10(min_cutoff)) // 3) * 3
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

    target_sites: NDArray

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
            shot_index: int = -1,
            target_sites: Optional[Union[Sequence[int], NDArray]] = None,  # deprecate this parameter
    ):
        super().__init__(preproc_h5_path=preproc_h5_path, shot_index=shot_index)

        self.plot_config = plot_config or PlotConfig()
        self._load_processed_quantities(preproc_h5_path)
        if target_sites is not None:
            self.target_sites = np.asarray(target_sites)

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
            try:
                self.target_sites = np.asarray(f.attrs['target_array'][:], dtype=int).flatten()
            except KeyError:
                logger.info('Did not find `target_array` in h5 attributes, defaulting to empty list')
                self.target_sites = np.array([], dtype=int)

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

    @property
    def target_sites_mask(self):
        return np.isin(np.arange(self.n_sites), self.target_sites)

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
        """
        Pandas Series describing site occupancy across all shots and images in this run.

        Example
        -------
        ```
        shot  image  site
        0     0      0        True
                     1       False
                     2        True
                     3        True
                     4        True
                             ...
        999   1      45       True
                     46      False
                     47      False
                     48      False
                     49      False
        Name: occupancy, Length: 100000, dtype: bool
        ```
        """
        # ignoring typechecker pending pandas-stubs#1285
        mi = pd.MultiIndex.from_product(
            [range(self.shots_processed), range(self.n_images), range(self.n_sites)],  # type: ignore
            sortorder=0,
            names=[self.KEY_SHOT, self.KEY_IMAGE, self.KEY_SITE],
        )
        return pd.Series(data=self.site_occupancies.flatten(), index=mi, name='occupancy', dtype=bool)

    def scan_param_df(self) -> pd.DataFrame:
        """
        Pandas DataFrame containing experimental parameters corresponding to each shot,
        indexed by shot number. Only varied parameters are provided.

        Example
        -------
              ryd_456_duration
        shot
        0         0.000000e+00
        1         1.428571e-07
        2         2.857143e-07
        3         4.285714e-07
        4         5.714286e-07
        ...                ...
        995       6.428571e-06
        996       6.571429e-06
        997       6.714286e-06
        998       6.857143e-06
        999       7.000000e-06

        [1000 rows x 1 columns]
        """
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

    # ==========================
    # Move elsewhere in codebase
    # ==========================

    def _shot_mask_target_full(self) -> np.ndarray:
        """
        True for shots where all target sites are filled in image 1,
        regardless of whether extra atoms exist outside target.
        """
        if (not self.rearrangement) or (len(self.target_sites) == 0):
            return np.ones(self.shots_processed, dtype=bool)

        img1 = self.site_occupancies[:, 1, :]  # (shots, sites)
        tgt = np.asarray(self.target_sites, dtype=int)
        return np.all(img1[:, tgt] == 1, axis=1)

    def _extras_outside_target(self) -> np.ndarray:
        """
        For each shot, number of occupied sites outside target_sites in image 1.
        """
        if (not self.rearrangement) or (len(self.target_sites) == 0):
            return np.zeros(self.shots_processed, dtype=int)

        tgt_mask = np.zeros(self.n_sites, dtype=bool)
        tgt_mask[np.asarray(self.target_sites, dtype=int)] = True

        img1 = self.site_occupancies[:, 1, :]
        return img1[:, ~tgt_mask].sum(axis=1).astype(int)

    def _extra_sites_mask(self) -> np.ndarray:
        """
        (shots, sites) boolean: True where a site is occupied outside target (image 1).
        """
        tgt_mask = np.zeros(self.n_sites, dtype=bool)
        tgt_mask[np.asarray(self.target_sites, dtype=int)] = True
        img1 = self.site_occupancies[:, 1, :].astype(bool)
        return img1 & (~tgt_mask[None, :])

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

    def plot_rearrange_histagram(
        self,
        target_array,
        ax,
        plot_overlapping_histograms: bool = True,
        split_full_target_bar: bool = True,
    ):
        """
        Histogram of number of loaded target sites after rearrangement (image 1),
        plus optional breakdown of "full target but extra atoms exist outside".
        """
        success_rearrange, atom_count_in_target_list, n_rearrange_shots, _ = self.rearragne_statistics(target_array)

        atom_counts_all_shots = atom_count_in_target_list[0]
        n_target = len(target_array)
        n_shots = len(self.site_occupancies)

        ax.set_xlabel('Number of loaded target sites after rearrangement')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y')

        # Make sure target_sites aligns with the passed target_array
        self.target_sites = list(np.asarray(target_array, dtype=int))

        mask_exact = self._shot_mask_exact_rearrangement()   # perfect: target full AND no extras
        mask_full  = self._shot_mask_target_full()           # target full regardless of extras
        mask_full_with_extras = mask_full & (~mask_exact)    # target full BUT extras exist
        extras_per_shot = self._extras_outside_target()

        n_exact = int(mask_exact.sum())
        n_full = int(mask_full.sum())
        n_full_with_extras = int(mask_full_with_extras.sum())
        n_full_clean = n_full - n_full_with_extras

        # for extra-atom summary when it happens
        extras_when_happens = extras_per_shot[mask_full_with_extras]
        mean_extras = float(extras_when_happens.mean()) if extras_when_happens.size else 0.0
        max_extras = int(extras_when_happens.max()) if extras_when_happens.size else 0

        title_parts = [f'{n_target} target sites']
        if n_shots > 0 and n_rearrange_shots > 0:
            ratio = n_rearrange_shots / n_shots
            rate = success_rearrange / n_rearrange_shots
            title_parts.append(f'rearrange shot ratio: {ratio:.3f}, success rate: {rate:.3f}')
        elif n_rearrange_shots == 0 and n_shots > 0:
            title_parts.append('no rearrangement attempts made')
        else:
            title_parts.append('no shot data for title metrics')

        if n_shots > 0:
            frac_full = n_full / n_shots
            frac_full_extras = (n_full_with_extras / n_full) if n_full > 0 else 0.0
            title_parts.append(
                f'full target: {n_full}/{n_shots} ({frac_full:.3f}); '
                f'full+extras: {n_full_with_extras}/{n_full if n_full else 1} ({frac_full_extras:.3f})'
                #f'⟨extras|happens⟩={mean_extras:.2f}, max={max_extras}'
            )

        ax.set_title('\n'.join(title_parts))

        # -------------------------
        # plot histograms
        # -------------------------
        if plot_overlapping_histograms:
            atom_counts_rearrange_shots = atom_count_in_target_list[1]

            unique_all, counts_all = np.unique(atom_counts_all_shots, return_counts=True)
            unique_rearrange, counts_rearrange = np.unique(atom_counts_rearrange_shots, return_counts=True)

            bar_width_all = 0.8
            bar_width_rearrange = bar_width_all * 0.7

            x_all = unique_all.astype(int)
            x_rearrange = unique_rearrange.astype(int)

            if len(counts_all) > 0:
                ax.bar(x_all, counts_all, width=bar_width_all,
                    label=f'All Shots ({n_shots} shots)', alpha=0.5, color='skyblue')
            if len(counts_rearrange) > 0:
                ax.bar(x_rearrange, counts_rearrange, width=bar_width_rearrange,
                    label=f'Rearrange Attempts ({n_rearrange_shots} shots)', alpha=0.8, color='royalblue')

            if len(x_all) > 0:
                ax.set_xticks(x_all)

            # ----------------------------------------------------
            # split the n_target bar (All shots)
            # ----------------------------------------------------
            if split_full_target_bar and n_shots > 0:
                # Draw a stacked bar at x=n_target that represents:
                # bottom = full clean (exact), top = full with extras
                # Put it slightly offset so it doesn't completely hide your existing bar.
                x0 = int(n_target) + 0.25
                w = 0.35
                ax.bar([x0], [n_full_clean], width=w, alpha=0.95,
                    label='Full target (clean)', color='tab:green')
                ax.bar([x0], [n_full_with_extras], width=w, bottom=[n_full_clean], alpha=0.95,
                    label='Full target (+extras)', color='tab:orange')

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
        print(f"Full target shots: {n_full}")
        print(f"  - clean (exact): {n_full_clean}")
        print(f"  - full + extras: {n_full_with_extras}")

    def plot_extras_count_when_target_full(self, target_array, ax):
        self.target_sites = list(np.asarray(target_array, dtype=int))
        mask_exact = self._shot_mask_exact_rearrangement()
        mask_full  = self._shot_mask_target_full()
        mask_full_with_extras = mask_full & (~mask_exact)

        extras = self._extras_outside_target()[mask_full_with_extras]
        if extras.size == 0:
            ax.text(0.5, 0.5, "No shots with full target + extras", ha='center', va='center')
            ax.set_axis_off()
            return

        vals, cnts = np.unique(extras, return_counts=True)
        ax.bar(vals.astype(int), cnts)
        ax.set_xlabel('# extra atoms outside target (given target full)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Extras count | full target + extras (N={extras.size})')
        ax.set_xticks(vals.astype(int))
        ax.grid(axis='y')

    def _indices_to_spans(self, indices: np.ndarray):
        """
        Convert sorted unique integer indices into contiguous spans [start, end] (inclusive).
        Example: [2,3,4, 7,8] -> [(2,4), (7,8)]
        """
        if indices.size == 0:
            return []

        spans = []
        start = prev = int(indices[0])

        for v in indices[1:]:
            v = int(v)
            if v == prev + 1:
                prev = v
            else:
                spans.append((start, prev))
                start = prev = v

        spans.append((start, prev))
        return spans

    def plot_extras_where_when_target_full(self, target_array, ax, mark_target=True, neighbor_radius=1):
        target_array = np.asarray(target_array, dtype=int)
        self.target_sites = list(target_array)

        mask_exact = self._shot_mask_exact_rearrangement()
        mask_full  = self._shot_mask_target_full()
        mask_event = mask_full & (~mask_exact)

        extra_sites = self._extra_sites_mask()
        extra_sel = extra_sites[mask_event]

        if extra_sel.shape[0] == 0:
            ax.text(0.5, 0.5, "No shots with full target + extras", ha='center', va='center')
            ax.set_axis_off()
            return

        p_extra = extra_sel.mean(axis=0)

        x = np.arange(self.n_sites)
        ax.bar(x, p_extra, width=0.9, alpha=0.9)

        ax.set_xlabel("Site index")
        ax.set_ylabel("P(extra at site | target full + extras)")
        ax.set_title(f"Where extra atoms appear (1D), N={extra_sel.shape[0]} shots")
        ax.grid(axis='y')

        if mark_target and target_array.size > 0:
            offsets = np.arange(-neighbor_radius, neighbor_radius + 1, dtype=int)
            halo = (target_array[:, None] + offsets[None, :]).ravel()
            halo = np.unique(halo[(halo >= 0) & (halo < self.n_sites)])

            # MERGE into contiguous spans so continuous targets don't "overmark"
            spans = self._indices_to_spans(halo)

            # Keep the original matplotlib default color look
            halo_alpha = 0.12
            line_alpha = 0.35
            line_width = 1.0

            # Shade halo spans
            for a, b in spans:
                ax.axvspan(a - 0.5, b + 0.5, alpha=halo_alpha)  # default color

            # Mark actual target sites
            ytop = ax.get_ylim()[1]
            ax.vlines(target_array, ymin=0, ymax=ytop, alpha=line_alpha, linewidth=line_width)  # default color

            # --- Legend: use proxy artists so symbols show up ---
            halo_patch = Patch(facecolor='C0', alpha=halo_alpha, label=f"Target ±{neighbor_radius}")
            target_line = Line2D([0], [0], color='C0', alpha=line_alpha, linewidth=1.5, label="Target sites")
            ax.legend(handles=[halo_patch, target_line], loc="upper right")

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

    # ====================
    # Correlation analysis
    # ====================

    def _plot_nmer_kexcited_population(
        self,
        ax,
        indep_var,
        nmer_size: int,
        require_exact_rearrangement: bool = True,
        explicit_groups: Optional[Sequence[Sequence[int]]] = None,
        parity_postselect: Optional[str] = None,  # None | "even" | "odd"
        save_data: bool = False,
    ):
        """
        Plot P(k excited) for k=0..nmer_size, where "excited" means occupied in image-3.

        If parity_postselect is "even" or "odd", we keep only shots with that parity
        (k even/odd) and renormalize probabilities within the kept subset.
        """

        if parity_postselect not in (None, "even", "odd"):
            raise ValueError('parity_postselect must be one of: None, "even", "odd".')

        if not self.rearrangement:
            raise ValueError("Requires rearrangement=True and 3 images (0,1,2).")
        if self.n_images < 3:
            raise ValueError("Expected 3 images (indices 0,1,2).")
        if nmer_size < 2:
            raise ValueError("nmer_size must be >= 2.")

        # Shot mask
        if require_exact_rearrangement:
            mask = self._shot_mask_exact_rearrangement()
            if not mask.any():
                ax.text(0.5, 0.5, "No exact-matching rearranged shots",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                return
        else:
            mask = np.ones(self.shots_processed, dtype=bool)

        # Build groups
        if explicit_groups is not None:
            groups = [tuple(map(int, g)) for g in explicit_groups]
            if any(len(g) != nmer_size for g in groups):
                raise ValueError("All explicit_groups must have length == nmer_size.")
        else:
            ts = list(map(int, self.target_sites))
            n_groups = len(ts) // nmer_size
            groups = [tuple(ts[i*nmer_size:(i+1)*nmer_size]) for i in range(n_groups)]

        if len(groups) == 0:
            raise ValueError("No n-mer groups could be formed (provide explicit_groups or more target_sites).")

        img_after_rearr   = self.site_occupancies[mask, 1, :].astype(bool, copy=False)
        img_after_science = self.site_occupancies[mask, 2, :].astype(bool, copy=False)

        xvals = self.current_params[mask, 0]
        x_arr = np.asarray(indep_var)
        use_isclose = (np.issubdtype(xvals.dtype, np.floating) or np.issubdtype(x_arr.dtype, np.floating))

        K = nmer_size
        counts_k = np.zeros((K + 1, x_arr.size), dtype=float)
        N = np.zeros(x_arr.size, dtype=float)  # denominator AFTER any post-selection

        for g in groups:
            g = np.asarray(g, dtype=int)

            # require cluster loaded at start of science
            loaded = np.all(img_after_rearr[:, g], axis=1)

            # k excited from image-3
            k_exc = np.sum(img_after_science[:, g], axis=1).astype(int)

            # parity mask per shot (only if post-selecting)
            if parity_postselect is None:
                parity_mask = np.ones_like(loaded, dtype=bool)
            elif parity_postselect == "even":
                parity_mask = (k_exc % 2 == 0)
            else:  # "odd"
                parity_mask = (k_exc % 2 == 1)

            for i, xv in enumerate(x_arr):
                bin_idx = np.isclose(xvals, xv) if use_isclose else (xvals == xv)

                # apply all selection conditions
                sel = bin_idx & loaded & parity_mask
                n_sel = int(sel.sum())
                if n_sel == 0:
                    continue

                # denominator is number of KEPT trials
                N[i] += n_sel

                kk = k_exc[sel]
                h = np.bincount(kk, minlength=K+1)
                counts_k[:, i] += h

        def division(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=(b > 0))

        def prop_and_std(k, n):
            # Laplace-smoothed posterior mean + beta std
            a = k + 1.0
            b = (n - k) + 1.0
            p = division(a, a + b)
            var = division(a * b, (a + b) ** 2 * (a + b + 1.0))
            return p, np.sqrt(var)

        p_k = np.full_like(counts_k, np.nan, dtype=float)  # shape (K+1, len(x))
        e_k = np.full_like(counts_k, np.nan, dtype=float)

        for k in range(K + 1):
            p_k[k, :], e_k[k, :] = prop_and_std(counts_k[k, :], N)

        # Plot
        for k in range(K + 1):
            # hide the parity-forbidden curves to avoid confusing legends
            if parity_postselect == "even" and (k % 2 == 1):
                continue
            if parity_postselect == "odd" and (k % 2 == 0):
                continue

            pk = p_k[k, :]
            ek = e_k[k, :]
            label = f"{k} excited"
            if parity_postselect is not None:
                label += f" (postsel {parity_postselect})"
            ax.errorbar(x_arr, pk, yerr=ek, label=label, **self.plot_config.errorbar_kw)

        ax.set_xlabel(self.params[0].axis_label, fontsize=self.plot_config.label_font_size)
        ax.set_ylabel(f"{nmer_size}-mer population", fontsize=self.plot_config.label_font_size)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.tick_params(axis="both", which="major", labelsize=self.plot_config.label_font_size)

        if save_data:
            parity_tag = "nopostsel" if parity_postselect is None else f"{parity_postselect}_parity"
            save_filename = f"ramsey_{nmer_size}mer_kexcited_{parity_tag}.npz"

            file_path = os.path.join(f"{self.folder_path}/", save_filename)

            np.savez(
                file_path,
                x_arr=x_arr,
                p_k=p_k,              # shape (K+1, len(x))
                e_k=e_k,
                counts_k=counts_k,
                N=N,
                nmer_size=nmer_size,
                parity_postselect=parity_postselect,
                groups=np.array(groups, dtype=object),
            )

            print(f"Saved N-mer population data to:\n{file_path}")

        # return x_arr, p_k, e_k, counts_k, N, groups
        # file_path = os.path.join(f"{self.folder_path}/", '2026-02-12-0150_ramsey_dimer_data.npz')
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
                ncols=1,
                nrows=2,
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

        gb = df.groupby([param.name for param in self.params])
        unitstr = self.params[0].unit
        survival_df = self.dataframe_survival(gb)

        indep_var = survival_df.index
        offset, xscale, scaled_unit = find_offset_and_scale(indep_var, unitstr)

        xlabel = self.params[0].axis_label(unit=scaled_unit)
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
            elif fit_type == 'rabi_oscillation':
                popt, pcov = self.fit_rabi_oscillation(indep_var, survival_rates, sigma=survival_rate_errs)
                upopt = uncertainties.correlated_values(popt, pcov)
                x_plot = np.linspace(np.min(indep_var), np.max(indep_var), 1000)
                ax_plot.plot(x_plot, self.rabi_model(x_plot, *popt), color='r', label=fR'$\Omega/2\pi = {upopt[1]/(2*np.pi*1e6):SL}$ MHz, $T_2 = {1e6*upopt[3]:SL} \mu s$')
                ax_plot.legend(fontsize='x-small')
            elif fit_type == 'rabispec':
                popt, pcov = self.fit_rabispec(indep_var, survival_rates, sigma=survival_rate_errs, peak_direction=-1)
                upopt = uncertainties.correlated_values(popt, pcov)

                freq_unit = self.params[0].unit
                label = textwrap.dedent(fR'''\
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
            self._plot_nmer_kexcited_population(ax_pairs, indep_var, 2, require_exact_rearrangement, parity_postselect="even")

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

            x_offset, xscale, scaled_xunit = find_offset_and_scale(cols, self.params[cols.name].unit)
            y_offset, yscale, scaled_yunit = find_offset_and_scale(ind, self.params[ind.name].unit)

            subplotspec = ax.get_subplotspec()
            if subplotspec is None:
                raise ValueError
            if subplotspec.is_last_row():
                ax.set_xlabel(
                    self.params[cols.name].axis_label(unit=scaled_xunit),
                    fontsize=self.plot_config.label_font_size,
                )
            if subplotspec.is_first_col():
                ax.set_ylabel(
                    self.params[ind.name].axis_label(unit=scaled_yunit),
                    fontsize=self.plot_config.label_font_size,
                )

            if subplotspec.is_first_row() and len(cross_section_info) > 0:
                cross_section_str = '; '.join(
                    f'{param.name} = {varval} {param.unit}'
                    for param, varval in cross_section_info.items()
                )
                ax.set_title(f'(at {cross_section_str})')
            return ax.pcolormesh(cols / xscale, ind / yscale, data, shading='nearest')

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

        ax.plot(
            run_time_series,
            unp.nominal_values(agg),
            **self.plot_config.plot_kw,
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

        ax.plot(
            loading_rates_unc.index,
            unp.nominal_values(loading_rates_unc),
            **self.plot_config.plot_kw,
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
            **self.plot_config.plot_kw,
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

        initial_atoms = self.initial_atoms_array
        # site_occupancies is of shape (num_shots, num_images, num_atoms)
        # axis=1 corresponds to the before/after tweezer images
        # multiplying along this axis gives 1 for (1, 1) (= survived atoms) and 0 otherwise
        surviving_atoms = initial_atoms * self.surviving_atoms_array

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
        print('self.target_sites:', self.target_sites)
        print('survival_rates shape:', survival_rates[self.target_sites, :, :].shape)
        if self.rearrangement:
            survival_rates = survival_rates[self.target_sites, :, :]

        return unique_params, survival_rates

    # TODO: merge this into plot_survival_rate_by_site
    def plot_survival_rate_by_site_2d(
            self,
            ax: Optional[Axes] = None,
            plot_grouped_averaged: bool = False,
            plot_targets_only: bool = None, # if not specified, will be set to self.rearrangement
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

        if plot_targets_only is None :
            plot_targets_only = self.rearrangement

        if plot_grouped_averaged:
            n_groups, averaged_data = self.group_data(survival_rates_matrix, group_size = 10)
            # 2D plot, group averaged
            pm = ax.pcolormesh(
                unique_params,
                np.arange(n_groups),
                averaged_data,
            )
        # elif plot_targets_only:
        #     # only plot rearrangement target sites
        #     print(self.target_sites)
        #     pm = ax.pcolormesh(
        #         unique_params,
        #         np.arange(len(self.target_sites)),
        #         survival_rates_matrix[self.target_sites, :],
        #     )
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

        unique_params_smooth = np.linspace(unique_params.min(), unique_params.max(), 1000)

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
                        ax.plot(unique_params_smooth, self.rabi_model(unique_params_smooth, *params_opt), 'r-', label='Fit')

                        annotation_text = (
                            f'p-p Ampl: {A_fit*2:.3f}\n'
                            f'Ω: {Omega_fit / 1e6 / (2*np.pi):.3f} MHz\n'
                            f'Phase: {phi_fit:.2f} rad\n'
                            f'T₂*: {T2_fit * 1e6:.2f} µs'
                        )
                        ax.annotate(annotation_text,
                                    xy=(0.02, 0.05), xycoords='axes fraction',
                                    fontsize=9, ha='left', va='bottom')
                    except Exception:
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
