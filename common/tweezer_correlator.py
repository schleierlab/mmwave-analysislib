from __future__ import annotations

from typing import Optional, Sequence
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig
from analysislib.common.typing import StrPath


class TweezerCorrelator(TweezerStatistician):
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
    
    def _shot_mask_exact_rearrangement_selected_sites(self) -> np.ndarray:
        """Boolean mask where selected sites match exactly at image 1."""
        if not self.rearrangement:
            return np.ones(self.shots_processed, dtype=bool)
        
        # Create a boolean pattern for selected sites
        selected_pattern = np.zeros(self.n_sites, dtype=bool)
        selected_pattern[np.asarray(self.selected_sites, dtype=int)] = True
        
        # Compare with image 1
        img1 = self.site_occupancies[:, 1, :]  # (shots, sites)
        return np.all(img1 == selected_pattern[None, :], axis=1)
    
    # def _combine_shot_masks(
    #         self,
    #         shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
    #         require_exact_rearrangement: bool = False,
    # ) -> Optional[np.ndarray | Sequence[np.ndarray]]:
    #     """Combine rearrangement mask with optional additional shot mask(s)."""
    #     if require_exact_rearrangement:
    #         rearr_mask = self._shot_mask_exact_rearrangement_selected_sites()
    #     else:
    #         rearr_mask = None
        
    #     if rearr_mask is None:
    #         return shot_mask
    #     elif shot_mask is None:
    #         return rearr_mask
    #     elif isinstance(shot_mask, (list, tuple)):
    #         # Multiple masks (list/tuple): combine each with rearrangement mask
    #         return [rearr_mask & mask for mask in shot_mask]
    #     elif isinstance(shot_mask, np.ndarray) and shot_mask.ndim == 2:
    #         # Multiple masks (2D array): combine each row with rearrangement mask
    #         return [rearr_mask & shot_mask[i] for i in range(shot_mask.shape[0])]
    #     else:
    #         # Single mask: combine with rearrangement mask
    #         return rearr_mask & shot_mask
    
    def extract_bitstrings(
            self,
            image_index: int = -1,
            shot_mask: Optional[np.ndarray | Sequence[np.ndarray]] = None,
            require_exact_rearrangement: bool = False,
    ) -> NDArray:
        """Extract survival bitstrings for selected sites (site[0] is MSB).
        
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
        combined_mask = shot_mask#self._combine_shot_masks(shot_mask, require_exact_rearrangement)
        
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
    
    def _bitstring_to_tuple(self, bitstring_row: NDArray) -> tuple:
        """Convert bitstring array to hashable tuple."""
        return tuple(int(x) for x in bitstring_row)
    
    def _tuple_to_bitstring(self, bitstring_tuple: tuple) -> str:
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
            title_suffix: str = "",
            group_by_magnetization: bool = False,
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
                            errors_curves[bs_tuple][param_idx] = np.mean(errors)/np.sqrt(shot_mask.shape[0])
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
        colors = plt.cm.bwr(np.linspace(0, 1, len(all_bitstrings)))

        # if group_by_magnetization:
        #     for mag in range(np.size(all_bitstrings[0])):

        
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


