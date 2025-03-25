from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Optional, cast
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

import h5py
import numpy as np
import yaml

from .image_preprocessor import ImagePreprocessor
from .analysis_config import kinetix_system
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
    load_type : str, default='lyse'
        Type of data loading to use
    h5_path : Optional[str], default=None
        Path to H5 file for data loading
    """

    MULTISHOT_PATH: ClassVar[Path] = Path(R'X:\userlib\analysislib\scripts\multishot')

    ROI_CONFIG_PATH: ClassVar[Path] = MULTISHOT_PATH / 'roi_config.yml'

    atom_roi: ROI
    background_roi: ROI
    site_rois: Sequence[ROI]
    threshold: float


    def __init__(
            self,
            load_type: str = 'lyse',
            h5_path: str = None):
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

        self.atom_roi, self.site_rois = self._load_rois_from_yaml()
        self.threshold = self._load_threshold_from_yaml()

    @property
    def n_sites(self):
        return len(self.site_rois)

    def _load_rois_from_yaml(self):
        # Get y-coordinates from globals
        try:
            atom_roi_ymin, atom_roi_height = self.globals["tw_kinetix_roi_row"]
            atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
        except KeyError:
            raise KeyError('tw_kinetix_roi_row not found in globals')

        with open(self.ROI_CONFIG_PATH) as stream:
            loaded_yaml = yaml.safe_load(stream)

        site_roi_arr = loaded_yaml['site_rois']
        atom_roi_xlims = loaded_yaml['atom_roi_xlims']

        return (
            ROI.from_roi_xy(atom_roi_xlims, atom_roi_ylims),
            [ROI.from_roi_xy(*pairpair) for pairpair in site_roi_arr]
        )

    def _load_threshold_from_yaml(self):
        """Load threshold value from file or use default."""
        with open(self.ROI_CONFIG_PATH) as stream:
            return yaml.safe_load(stream)['threshold']

    def process_shot(self):
        images = [
            Image(
                exposure,
                background=self.exposures[-1],
                yshift=self.atom_roi.ymin,
            )
            for exposure in self.exposures[:-1]
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

        cmap = 'bone'
        norm = Normalize(vmin=10, vmax=vmax)
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
                for j, roi in enumerate(self.site_rois):
                    ax.add_patch(roi.patch(
                        edgecolor=('yellow' if self.site_occupancies[i, j] else 'red'),
                    ))

        fig.colorbar(ScalarMappable(norm, cmap='bone'), ax=axs, label='Counts')
