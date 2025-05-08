import importlib.resources
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Optional, cast

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .image_preprocessor import ImagePreprocessor
from .analysis_config import kinetix_system
from .image import Image, ROI
from analysislib import multishot


class TweezerPreprocessor(ImagePreprocessor):
    """Analysis class for tweezer imaging data.

    This class handles the analysis of tweezer imaging data, including:
    - Loading and preprocessing images
    - Background subtraction
    - ROI-based analysis
    - Threshold-based atom detection

    Parameters
    ----------
    load_type : str, default='lyse'
        Type of data loading to use
    h5_path : Optional[str], default=None
        Path to H5 file for data loading
    """

    ROI_CONFIG_PATH: ClassVar[Path] = importlib.resources.files(multishot) / 'roi_config.yml'
    PROCESSED_RESULTS_FNAME: ClassVar[str] = 'tweezer_preprocess.h5'
    DEFAULT_PARAMS_PATH: ClassVar[Path] = Path('X:/userlib/labscriptlib/defaults.yml')

    atom_roi: ROI
    background_roi: ROI
    site_rois: Sequence[ROI]
    threshold: float


    def __init__(
            self,
            load_type: str = 'lyse',
            h5_path: str = None,
            use_averaged_background: bool = False
        ):
        """Initialize TweezerAnalysis with analysis configuration.

        Parameters
        ----------
        load_type : str, default='lyse'
            Type of loading to perform, 'lyse' or 'h5'. 'h5' requires h5_path to be specified.
        h5_path : str, optional
            Path to h5 file to load, only used if load_type='h5'
        """

        # Initialize parent class first
        super().__init__(
            imaging_setup=kinetix_system,
            load_type=load_type,
            h5_path=h5_path
        )

        self.atom_roi, self.site_rois = TweezerPreprocessor._load_rois_from_yaml(self.ROI_CONFIG_PATH, self._load_ylims_from_globals())
        self.threshold, self.site_thresholds = self._load_threshold_from_yaml(self.ROI_CONFIG_PATH)
        if use_averaged_background:
            average_background_overwrite_path = Path(r'X:\userlib\analysislib\multishot\avg_shot_bkg.npy')
            bkg = np.load(average_background_overwrite_path)
        else:
            bkg = self.exposures[-1]

        self.images = [
            Image(
                exposure,
                background=bkg,
                yshift=self.atom_roi.ymin,
            )
            for exposure in self.exposures[:-1]
        ]

    @property
    def n_sites(self):
        return len(self.site_rois)

    def _load_ylims_from_globals(self):
        """Load the atom ROI y-coordinates from globals. Sometimes they change...
        # TODO: further explanation

        Returns
        -------
        atom_roi_ylims : list
            List of y-coordinates for the atom ROI.
        """
        try:
            atom_roi_ymin, atom_roi_height = self.globals["kinetix_roi_row"]
            atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
        except KeyError:
            try:
                default_params = self._load_default_params_from_yaml(self.DEFAULT_PARAMS_PATH)
                atom_roi_ymin, atom_roi_height  = np.array(eval(default_params["Tweezers"]["kinetix_roi_row"]['value']))
                atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
            except KeyError:
                raise KeyError('kinetix_roi_row not found in globals')
        return atom_roi_ylims

    @property
    def target_array(self):
        try:
            target_array = self.globals['TW_target_array']
        except KeyError:
            try:
                target_array = self.default_params['TW_target_array']
            except KeyError:
                raise KeyError('no TW_target_array find in both globals and default values')
        return target_array

    @staticmethod
    def _load_default_params_from_yaml(defaul_params_path: Path):
        """
        Load default parameters from YAML file.

        Parameters
        ----------
        defaul_params_path : Path
            Path to the YAML file containing the default parameters.

        Returns
        -------
        default_params : dict
            Dictionary of default parameters.
        """
        with defaul_params_path.open('rt') as stream:
            default_params = yaml.safe_load(stream)

        return default_params

    @staticmethod
    def _load_rois_from_yaml(roi_config_path: Path, atom_roi_ylims):
        """Load site ROIs from YAML file.
        Want this to be static so that it can be used by TweezerFinder.

        Parameters
        ----------
        roi_config_path : Path
            Path to the YAML file containing the ROIs.
        atom_roi_ylims : list
            List of y-coordinates for the atom ROI.

        Returns
        -------
        atom_roi : ROI
            ROI object for the atom region.
        site_rois : list
            List of ROI objects for each site.
        """
        with roi_config_path.open('rt') as stream:
            loaded_yaml = yaml.safe_load(stream)

        site_roi_arr = loaded_yaml['site_rois']
        atom_roi_xlims = loaded_yaml['atom_roi_xlims']

        return (
            ROI.from_roi_xy(atom_roi_xlims, atom_roi_ylims),
            [ROI.from_roi_xy(*pairpair) for pairpair in site_roi_arr]
        )

    @staticmethod
    def _load_threshold_from_yaml(roi_config_path: Path):
        """Load threshold value from file or use default.

        Want this to be static so that it can be used by TweezerFinder.

        Parameters
        ----------
        roi_config_path : Path
            Path to the YAML file containing the ROIs.

        Returns
        -------
        threshold : float
            Threshold value for atom detection.
        """
        with roi_config_path.open('rt') as stream:
            global_threshold = yaml.safe_load(stream)['threshold']
        with roi_config_path.open('rt') as stream:
            site_thresholds = yaml.safe_load(stream)['site_thresholds']

        return global_threshold, site_thresholds

    @staticmethod
    def dump_to_yaml(
        site_rois: Sequence[ROI],
        atom_roi: ROI,
        global_threshold: float,
        site_thresholds: list[float],
        output_path: str,
    ) -> str:
        """
        Dump site ROIs to a YAML file in the same format as roi_config.yml.

        Want this to be static so that it can be used by TweezerFinder.

        Parameters
        ----------
        site_rois : Sequence[ROI]
            List of ROI objects for each site
        atom_roi : Optional[ROI], default=None
            ROI object for the atom region.
        global_threshold : float
            Global threshold value for atom detection.
        site_thresholds : list[float]
            List of site-specific threshold values for atom detection.
        output_path : str, optional
            Path to save the YAML file. If None, will save to 'roi_test.yml'
            in the same directory as the original roi_config.yml

        Returns
        -------
        str
            Path to the created YAML file
        """

        # Convert ROI objects to the format used in the YAML file
        site_rois_formatted = []
        for roi in site_rois:
            # Each site ROI is represented as [[xmin, xmax], [ymin, ymax]]
            site_rois_formatted.append([[roi.xmin, roi.xmax], [roi.ymin, roi.ymax]])

        # Get atom_roi_xlims and atom_roi_ylims from the atom_roi object
        atom_roi_xlims = [atom_roi.xmin, atom_roi.xmax]
        atom_roi_ylims = [atom_roi.ymin, atom_roi.ymax]

        # Determine the output path
        output_path = Path(output_path)

        # Format the YAML content manually to match the exact format of roi_config.yml
        yaml_content = "---\n"
        yaml_content += f"threshold: {global_threshold}\n"
        yaml_content += f"atom_roi_xlims: [{atom_roi_xlims[0]}, {atom_roi_xlims[1]}]\n"
        yaml_content += f"atom_roi_ylims: [{atom_roi_ylims[0]}, {atom_roi_ylims[1]}]\n"

        # Format the site ROIs
        yaml_lines = ["site_rois:"]
        for site in site_rois_formatted:
            yaml_lines.append(f"  - [[{site[0][0]}, {site[0][1]}],")
            yaml_lines.append(f"     [{site[1][0]}, {site[1][1]}]]")

        yaml_content += "\n".join(yaml_lines)
        yaml_content += "\n"

        # Format the site thresholds
        yaml_lines = ["site_thresholds:"]
        for threshold in site_thresholds:
            yaml_lines.append(f"  - {threshold}")

        yaml_content += "\n".join(yaml_lines)

        # Write the YAML file
        with output_path.open('w') as stream:
            stream.write(yaml_content)

        return str(output_path)

    def process_shot(self, use_global_threshold: bool = False):
        camera_counts = np.array([image.roi_sums(self.site_rois) for image in self.images])

        # Implement the thresholding to determine site occupancy
        if use_global_threshold: # means we use the same threshold for all sites
            print("Using global threshold =", self.threshold)
            self.site_occupancies = camera_counts > self.threshold
        else:
            print("Using site-specific thresholds...")
            self.site_occupancies = camera_counts > self.site_thresholds

        run_number = self.run_number
        fname = self.h5_path.with_name(self.PROCESSED_RESULTS_FNAME)
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.attrs['n_runs'] = self.n_runs
                f.create_dataset('camera_counts', data=camera_counts[np.newaxis, ...], maxshape=(None, 10, 100))  # shape: (n_shots, n_images, n_sites)
                f.create_dataset('site_occupancies', data=self.site_occupancies[np.newaxis, ...], maxshape=(None, 10, 100))
                f['site_occupancies'].attrs['threshold'] = self.threshold
                f.create_dataset('site_rois', data=ROI.toarray(self.site_rois))
                f['site_rois'].attrs['fields'] = ['xmin', 'xmax', 'ymin', 'ymax']

                # save parameters from runmanager globals
                f.create_dataset(
                    'current_params',
                    data=self.current_params[np.newaxis, ...],
                    maxshape=(self.n_runs, len(self.current_params)),
                    chunks = True,
                )
                param_list = []
                for key in self.params.keys():
                    param_list.append([key, self.params[key][1], self.params[key][0]])
                f.create_dataset('params', data=param_list)
        else:
            with h5py.File(fname, 'a') as f:
                f['camera_counts'].resize(run_number + 1, axis=0)
                f['camera_counts'][run_number] = camera_counts

                f['site_occupancies'].resize(run_number + 1, axis=0)
                f['site_occupancies'][run_number] = self.site_occupancies

                # save parameters from runmanager globals
                f['current_params'].resize(run_number + 1, axis=0)
                f['current_params'][run_number] = self.current_params

        return fname

    def show_image(
            self,
            roi_patches: bool = True,
            site_index: bool = True,
            fig: Optional[Figure] = None,
            vmax: Optional[int] = 70,
            cmap: str = 'viridis', #'bone'
    ):
        if fig is None:
            fig, axs = plt.subplots(
                nrows=len(self.images),
                ncols=1,
            )
        else:
            axs = fig.subplots(nrows=2, ncols=1)

        norm = Normalize(vmin=0, vmax=vmax)
        for i, image in enumerate(self.images):
            ax = cast(Axes, axs[i])
            image.imshow_view(
                self.atom_roi,
                ax=ax,
                cmap=cmap,
                norm=norm,
            )
            ax.set_title(f'Image {i}')
            if roi_patches:
                patches = tuple(
                    roi.patch(edgecolor=('yellow' if self.site_occupancies[i, j] else 'red'), alpha=0.6)
                    for j, roi in enumerate(self.site_rois)
                )
                collection = PatchCollection(patches, match_original=True)
                ax.add_collection(collection)
                text_kwargs = {
                    'color':'red',
                    'fontsize':'small',
                    }
                if site_index:
                    [ax.annotate(
                        str(j), # The site index to display
                        xy=(roi.xmin, roi.ymin - 5), # Position of the text
                        **text_kwargs
                        )
                    # Iterate through sites, but only annotate if j is a multiple of 5
                    for j, roi in enumerate(self.site_rois) if j % 5 == 0]

            fig.suptitle(
                self.h5_path,
            )

        fig.colorbar(ScalarMappable(norm, cmap=cmap), ax=axs, label='Counts')
