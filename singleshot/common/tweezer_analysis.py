from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Optional, cast
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import assert_never

import h5py
import numpy as np
import yaml

from .image_preprocessor import ImagePreprocessor
from .analysis_config import TweezerAnalysisConfig
from .image import Image, ROI


class TweezerPreprocessor(ImagePreprocessor):
    """Analysis class for tweezer imaging data.

    This class handles the analysis of tweezer imaging data, including:
    - Loading and preprocessing images
    - Background subtraction
    - ROI-based analysis
    - Threshold-based atom detection

    Parameters
    ----------
    config : TweezerAnalysisConfig
        Configuration object containing all analysis parameters
    load_type : str, default='lyse'
        Type of data loading to use
    h5_path : Optional[str], default=None
        Path to H5 file for data loading
    """

    MULTISHOT_PATH: ClassVar[Path] = Path(R'X:\userlib\analysislib\scripts\multishot')
    THRESHOLD_PATH: ClassVar[Path] = MULTISHOT_PATH / 'th.npy'

    ROI_X_PATH: ClassVar[Path] = MULTISHOT_PATH / 'roi_x.npy'
    SITE_ROI_X_PATH: ClassVar[Path] = MULTISHOT_PATH / 'site_roi_x.npy'
    SITE_ROI_Y_PATH: ClassVar[Path] = MULTISHOT_PATH / 'site_roi_y.npy'

    ROI_CONFIG_PATH: ClassVar[Path] = MULTISHOT_PATH / 'roi_config.yml'

    atom_roi: ROI
    background_roi: ROI
    site_rois: Sequence[ROI]
    threshold: float


    def __init__(
            self,
            config: TweezerAnalysisConfig,
            load_type: str = 'lyse',
            h5_path: str = None):
        """Initialize TweezerAnalysis with analysis configuration.

        Parameters
        ----------
        config : TweezerAnalysisConfig
            Configuration object containing all analysis parameters including:
            - imaging_system: ImagingSystem configuration
            - method: Background subtraction method
            - bkg_roi_x: Background ROI x-coordinates
            - load_roi: Whether to load ROIs from files
            - roi_config_path: Path to ROI config YAML
            - load_threshold: Whether to load threshold from file
            - threshold: Optional threshold value
        load_type : str, default='lyse'
            Type of loading to perform
            # TODO: what are the options for load_type?
        h5_path : str, optional
            Path to h5 file to load
        """

        # Initialize parent class first
        super().__init__(
            imaging_setup=config.imaging_system,
            load_type=load_type,
            h5_path=h5_path
        )

        # Store config
        self.config = config

        self.atom_roi, self.background_roi, self.site_rois = self.load_rois_from_yaml()
        self.threshold = self.load_threshold_from_yaml()

    @property
    def n_sites(self):
        return len(self.site_rois)

    # TODO this return signature is a mess -- can we simplify?
    # TODO: Nolan: I can delete this?
    # def load_rois(self) -> tuple[ROI, ROI, list[ROI]]:
    #     """Load ROI and site ROI either from file or from provided configuration.

    #     Parameters
    #     ----------
    #     roi_config_path : str
    #         Path to YAML configuration file containing ROI specifications.
    #         When load_roi=False:
    #         - Must contain 'roi_x' key with [start, end] coordinates
    #         - Must contain 'site_roi' section with 'site_roi_x' and 'site_roi_y'
    #     bkg_roi_x : list[int]
    #         X-coordinates [start, end] for background region.
    #         This is passed directly from __init__ rather than loaded from YAML.
    #     load_roi : bool
    #         If True, load ROIs from standard file locations
    #         If False, use ROIs from YAML config

    #     Returns
    #     -------
    #     atom_roi : ROI
    #         [[x_min, x_max], [y_min, y_max]] for atom ROI
    #     background_roi : ROI
    #         [[x_min, x_max], [y_min, y_max]] for background ROI
    #     site_roi : list[ROI]
    #         [site_roi_x, site_roi_y] for site-specific ROIs
    #     """
    #     # Get y-coordinates from globals
    #     try:
    #         roi_y = self.globals["tw_kinetix_roi_row"]
    #     except KeyError:
    #         print("Warning: tw_kinetix_roi_row not found in globals, using full camera height")
    #         roi_y = [0, self.imaging_setup.camera.image_size]

    #     if self.config.load_roi:
    #         # Load ROIs from standard .npy files
    #         try:
    #             roi_x = np.load(self.ROI_X_PATH).tolist()
    #             site_roi_x = np.load(self.SITE_ROI_X_PATH)
    #             site_roi_y = np.load(self.SITE_ROI_Y_PATH)
    #         except FileNotFoundError as e:
    #             raise FileNotFoundError(f"Could not load ROI files from {self.MULTISHOT_PATH}: {e}")

    #         # Adjust site_roi_x relative to roi_x
    #         site_roi_x = site_roi_x - roi_x[0]

    #         # Add bkg site roi
    #         site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], np.array(site_roi_x)])
    #         site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) - 10], np.array(site_roi_y)])

    #     else:
    #         # When not loading from files, get the ROIs from the config
    #         if self.config.roi_x is None:
    #             raise ValueError("When load_roi is False, the config must include 'roi_x' coordinates")
    #         if self.config.site_roi is None:
    #             raise ValueError("When load_roi is False, the config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

    #         # Load ROIs from config
    #         roi_x = self.config.roi_x
    #         site_roi_x = np.array(self.config.site_roi[0])
    #         site_roi_y = np.array(self.config.site_roi[1])

    #     # Return ROIs in the format expected by the class

    #     # swap axes so that shape is (n_sites, 2[x, y], 2[min, max])
    #     site_roi_arr = np.transpose([site_roi_x, site_roi_y], (1, 0, 2))

    #     return (
    #         ROI.from_roi_xy(roi_x, roi_y),
    #         ROI.from_roi_xy(self.config.bkg_roi_xlims, roi_y),
    #         [ROI.from_roi_xy(*pairpair) for pairpair in site_roi_arr]
    #     )

    def load_rois_from_yaml(self):
        # Get y-coordinates from globals
        try:
            atom_roi_ymin, atom_roi_height = self.globals["tw_kinetix_roi_row"]
            atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
        except KeyError:
            raise KeyError('tw_kinetix_roi_row not found in globals')

        if self.config.load_roi:
            with open(self.ROI_CONFIG_PATH) as stream:
                try:
                    loaded_yaml = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            site_roi_arr = loaded_yaml['site_rois']
            atom_roi_xlims = loaded_yaml['atom_roi_xlims']

            # # Add bkg site roi
            # site_roi_x = np.concatenate([
            #     [np.min(site_roi_x, axis=0)],
            #     np.array(site_roi_x),
            # ])
            # site_roi_y = np.concatenate([
            #     [np.min(site_roi_y, axis=0) - 10],
            #     np.array(site_roi_y),
            # ])

        else:
            # When not loading from files, get the ROIs from the config
            if self.config.roi_x is None:
                raise ValueError("When load_roi is False, the config must include 'roi_x' coordinates")
            if self.config.site_roi is None:
                raise ValueError("When load_roi is False, the config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

            # Load ROIs from config
            atom_roi_xlims = self.config.roi_x
            site_roi_x = np.array(self.config.site_roi[0])
            site_roi_y = np.array(self.config.site_roi[1])
            site_roi_arr = np.transpose([site_roi_x, site_roi_y], (1, 0, 2))

        return (
            ROI.from_roi_xy(atom_roi_xlims, atom_roi_ylims),
            ROI.from_roi_xy(self.config.bkg_roi_xlims, atom_roi_ylims),
            [ROI.from_roi_xy(*pairpair) for pairpair in site_roi_arr]
        )

    # TODO: Nolan: I can delete this?
    # def load_threshold(self) -> float:
    #     """Load threshold value from file or use default."""
    #     if self.config.load_threshold:
    #         try:
    #             threshold = np.load(self.THRESHOLD_PATH)[0]
    #             return threshold
    #         except FileNotFoundError as e:
    #             raise FileNotFoundError(f'Could not load threshold from {self.THRESHOLD_PATH}: file does not exist! {e}')

    #     default_threshold = self.config.threshold
    #     if default_threshold is None:
    #         raise ValueError(
    #             'When load_threshold is False, default_threshold must be provided'
    #         )

    #     return default_threshold

    def load_threshold_from_yaml(self):
        """Load threshold value from file or use default."""
        if self.config.load_threshold:
            try:
                with open(self.ROI_CONFIG_PATH) as stream:
                    try:
                        return yaml.safe_load(stream)['threshold']
                    except yaml.YAMLError as exc:
                        print(exc)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Could not load threshold from {self.THRESHOLD_PATH}: file does not exist! {e}')

        default_threshold = self.config.threshold
        if default_threshold is None:
            raise ValueError(
                'When load_threshold is False, default_threshold must be provided'
            )

        return default_threshold

    def process_shot(self):
        n_images = len(self.exposures) - 1

        name_stem = self.imaging_setup.camera.image_name_stem
        images = [
            Image(
                self.exposures[f'{name_stem}{i}'],
                background=self.exposures[f'{name_stem}{n_images}'],
                yshift=self.atom_roi.ymin,
            )
            for i in range(n_images)
        ]
        self.images = images

        camera_counts = np.array([image.roi_sums(self.site_rois) for image in images])
        self.site_occupancies = camera_counts > self.threshold

        run_number = self.run_number
        fname = Path(self.folder_path) / 'tweezer_preprocess.h5'
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.create_dataset('camera_counts', data=camera_counts[np.newaxis, ...], maxshape=(None, 10, 100))  # shape: (n_shots, n_images, n_sites)
                f.create_dataset('site_occupancies', data=self.site_occupancies[np.newaxis, ...], maxshape=(None, 10, 100))
                f['site_occupancies'].attrs['threshold'] = self.threshold
                f.create_dataset('site_rois', data=ROI.toarray(self.site_rois))
                f['site_rois'].attrs['fields'] = ['xmin', 'xmax', 'ymin', 'ymax']
        else:
            with h5py.File(fname, 'a') as f:
                f['camera_counts'].resize(run_number + 1, axis=0)
                f['camera_counts'][run_number] = camera_counts

                f['site_occupancies'].resize(run_number + 1, axis=0)
                f['site_occupancies'][run_number] = self.site_occupancies

        return fname

    def show_image(self, roi_patches: bool = True, fig: Optional[Figure] = None, vmax: Optional[int] = 70):
        if fig is None:
            fig, axs = plt.subplots(
                nrows=len(self.images),
                ncols=1,
            )
        else:
            axs = fig.subplots(nrows=2, ncols=1)

        for i, image in enumerate(self.images):
            ax = cast(Axes, axs[i])
            im = image.imshow_view(
                self.atom_roi, ax=ax,
                cmap='bone',
                vmin=10,
                vmax=vmax,
            )
            plt.colorbar(im, ax=ax, label='Counts')
            ax.set_title(f'Image {i}')
            if roi_patches:
                for j, roi in enumerate(self.site_rois):
                    ax.add_patch(roi.patch(
                        edgecolor=('yellow' if self.site_occupancies[i, j] else 'red'),
                    ))
