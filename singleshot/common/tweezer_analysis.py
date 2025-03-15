from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from typing_extensions import assert_never

import h5py
import numpy as np

from .image_preprocessor import ImagePreprocessor
from .analysis_config import PairPair, TweezerAnalysisConfig
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

    atom_roi: PairPair[int]
    background_roi: PairPair[int]
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

        self.atom_roi, self.background_roi, self.site_rois = self.load_rois()
        self.threshold = self.load_threshold()

    @property
    def n_sites(self):
        return len(self.site_rois)

    # TODO this return signature is a mess -- can we simplify?
    def load_rois(self) -> tuple[ROI, ROI, list[ROI]]:
        """Load ROI and site ROI either from file or from provided configuration.

        Parameters
        ----------
        roi_config_path : str
            Path to YAML configuration file containing ROI specifications.
            When load_roi=False:
            - Must contain 'roi_x' key with [start, end] coordinates
            - Must contain 'site_roi' section with 'site_roi_x' and 'site_roi_y'
        bkg_roi_x : list[int]
            X-coordinates [start, end] for background region.
            This is passed directly from __init__ rather than loaded from YAML.
        load_roi : bool
            If True, load ROIs from standard file locations
            If False, use ROIs from YAML config

        Returns
        -------
        atom_roi : list[list[int]]
            [[x_min, x_max], [y_min, y_max]] for atom ROI
        background_roi : list[list[int]]
            [[x_min, x_max], [y_min, y_max]] for background ROI
        site_roi : list[list[list[int]]]
            [site_roi_x, site_roi_y] for site-specific ROIs
        """
        # Get y-coordinates from globals
        try:
            roi_y = self.globals["tw_kinetix_roi_row"]
        except KeyError:
            print("Warning: tw_kinetix_roi_row not found in globals, using full camera height")
            roi_y = [0, self.imaging_setup.camera.image_size]

        if self.config.load_roi:
            # Load ROIs from standard .npy files
            try:
                roi_x = np.load(self.ROI_X_PATH).tolist()
                site_roi_x = np.load(self.SITE_ROI_X_PATH)
                site_roi_y = np.load(self.SITE_ROI_Y_PATH)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load ROI files from {self.MULTISHOT_PATH}: {e}")

            # Adjust site_roi_x relative to roi_x
            site_roi_x = site_roi_x - roi_x[0]

            # Add bkg site roi
            site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], np.array(site_roi_x)])
            site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) - 10], np.array(site_roi_y)])

        else:
            # When not loading from files, get the ROIs from the config
            if self.config.roi_x is None:
                raise ValueError("When load_roi is False, the config must include 'roi_x' coordinates")
            if self.config.site_roi is None:
                raise ValueError("When load_roi is False, the config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

            # Load ROIs from config
            roi_x = self.config.roi_x
            site_roi_x = np.array(self.config.site_roi[0])
            site_roi_y = np.array(self.config.site_roi[1])

        # Return ROIs in the format expected by the class

        # swap axes so that shape is (n_sites, 2[x, y], 2[min, max])
        site_roi_arr = np.transpose([site_roi_x, site_roi_y], (1, 0, 2))

        return (
            ROI.from_roi_xy(roi_x, roi_y),
            ROI.from_roi_xy(self.config.bkg_roi_x, roi_y),
            [ROI.from_roi_xy(*pairpair) for pairpair in site_roi_arr]
        )

    def load_threshold(self) -> float:
        """Load threshold value from file or use default."""
        if self.config.load_threshold:
            try:
                threshold = np.load(self.THRESHOLD_PATH)[0]
                return threshold
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Could not load threshold from {self.THRESHOLD_PATH}: file does not exist! {e}')

        default_threshold = self.config.threshold
        if default_threshold is None:
            raise ValueError(
                'When load_threshold is False, default_threshold must be provided'
            )

        return default_threshold

    def get_image_bkg_sub(self):
        """Get background-subtracted images.
        Returns
        -------
        atom_images : ndarray
            Array of atom images
        background_images : ndarray
            Array of background images
        sub_images : ndarray
            Array of background-subtracted images
        """
        images = self.images
        folder_path = Path(self.folder_path)

        # Set up paths for background images
        alternating_bkg_path = folder_path / 'alternating_bkg'
        last_bkg_sub_path = folder_path / 'last_bkg_sub'

        if self.config.method == 'alternating':
            if self.globals['mot_do_coil']:
                atom_images = images
                background_images = np.load(alternating_bkg_path)
                subtracted_images = atom_images - background_images
                np.save(last_bkg_sub_path, subtracted_images)
            else:
                background_images = images
                np.save(alternating_bkg_path, images)
                subtracted_images = np.load(last_bkg_sub_path)
                # load last background subtracted images
                # during background taking shot to make
                # sure there is something to plot
        elif self.config.method == 'average':
            atom_images = images
            background_images = np.load(folder_path / 'avg_shot_bkg.npy')
            subtracted_images = atom_images - background_images
            np.save(last_bkg_sub_path, subtracted_images)
        else:
            assert_never(self.config.method)

        return atom_images, background_images, subtracted_images

    def process_shot(self):
        n_images = len(self.images) - 1

        name_stem = self.imaging_setup.camera.image_name_stem
        images = [
            Image(
                self.images[f'{name_stem}{i}'],
                background=self.images[f'{name_stem}{n_images - 1}'],
            )
            for i in range(n_images)
        ]

        camera_counts = np.array([image.roi_sums(self.site_rois) for image in images])
        site_occupancies = camera_counts > 100  # self.threshold

        run_number = self.run_number
        fname = Path(self.folder_path) / 'tweezer_preprocess.h5'
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.create_dataset('camera_counts', data=camera_counts[np.newaxis, ...], maxshape=(None, 10, 100))  # shape: (n_shots, n_images, n_sites)
                f.create_dataset('site_occupancies', data=site_occupancies[np.newaxis, ...], maxshape=(None, 10, 100))
                f.create_dataset('site_rois', data=ROI.toarray(self.site_rois))
                f['site_rois'].attrs['fields'] = ['xmin', 'xmax', 'ymin', 'ymax']
        else:
            with h5py.File(fname, 'a') as f:
                f['camera_counts'].resize(run_number + 1, axis=0)
                f['camera_counts'][run_number] = camera_counts

                f['site_occupancies'].resize(run_number + 1, axis=0)
                f['site_occupancies'][run_number] = site_occupancies

        return fname
