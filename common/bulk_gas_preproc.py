from pathlib import Path
from typing import ClassVar, Literal, cast, Optional

import h5py
import numpy as np
import uncertainties
import uncertainties.unumpy as unumpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import pi

try:
    lyse
except:
    import lyse

from .analysis_config import BulkGasAnalysisConfig
from .image import Image, ROI
from .image_preprocessor import ImagePreprocessor


class BulkGasPreprocessor(ImagePreprocessor):
    """Analysis class for bulk gas imaging data.

    This class provides functionality for analyzing bulk gas imaging data, including
    ROI-based analysis, background subtraction, and atom number calculations.

    The class uses a configuration-based approach where all analysis parameters
    are specified through an BulkGasAnalysisConfig object, which includes imaging system
    setup, ROI definitions, and analysis parameters.
    """

    scattering_rate: ClassVar = 2 * pi * 5.2227e+6  # rad/s
    """ Cesium scattering rate, in radians / second """
    # TODO: for what transition? Where did you get this number?

    atoms_roi: ROI
    background_roi: ROI

    def __init__(
            self,
            config: BulkGasAnalysisConfig,
            load_type: str = 'lyse',
            h5_path: str = None
            ):
        """Initialize BulkGasAnalysis with analysis configuration.

        Parameters
        ----------
        config : BulkGasAnalysisConfig
            Configuration object containing all analysis parameters including:
            - imaging_system: ImagingSystem configuration
            - exposure_time: Imaging exposure time in seconds
            - atoms_roi: ROI for atoms [[x_min, x_max], [y_min, y_max]]
            - bkg_roi: ROI for background [[x_min, x_max], [y_min, y_max]]
            - method: Background subtraction method
        load_type : str, default='lyse'
            Type of loading to perform
             # TODO: what are the options for load_type?
        h5_path : str, optional
            Path to h5 file to load
        """
        if config.exposure_time is None:
            raise ValueError("exposure_time must be provided in BulkGasAnalysisConfig for bulk gas analysis")
        if config.atoms_roi is None or config.bkg_roi is None:
            raise ValueError("atoms_roi and bkg_roi must be provided in BulkGasAnalysisConfig for bulk gas analysis")

        # Initialize parent class first
        super().__init__(
            imaging_setup=config.imaging_system,
            load_type=load_type,
            h5_path=h5_path
        )

        # Store config
        self.analysis_config = config

        # Set class-specific attributes
        self.atoms_roi = config.atoms_roi
        self.background_roi = config.bkg_roi
        self.counts_per_atom = self.imaging_setup.counts_per_atom(
            self.scattering_rate,
            config.exposure_time,
        )

        atom_image, background_image = self.exposures
        self.image = Image(atom_image, background_image)

    def get_atom_number(
            self,
            method: Literal['sum', 'fit'] = 'sum',
            subtraction: Literal['simple', 'double'] = 'simple',
    ):
        """
        sum of counts in roi/counts_per_atom - average background atom per px* roi size

        Parameters
        ----------
        method: {'sum', 'fit'}
            Method for atom number calculation. If 'sum', just take the total counts
            and convert to an atom number. If 'fit', fit the observed image to a
            2D Gaussian and integrate the Gaussian to get the total counts.
        subtraction: {'simple', 'double'}
            Background subtraction method. If 'simple', do the naive thing.
            If 'double', use the background-subtracted image and further
            use a distant part of the image to estimate any further background
            in the already-background-subtracted image (which may arise from
            e.g. drifts in TA power.)

        Returns
        -------
        atom_number: float
        """
        if method == 'fit':
            raise NotImplementedError

        atom_counts = self.image.roi_sum(self.atoms_roi)
        if subtraction == 'double':
            area_ratio = self.atoms_roi.pixel_area / self.background_roi.pixel_area
            atom_counts -= self.image.roi_sum(self.background_roi) * area_ratio

        return atom_counts / self.counts_per_atom

    def get_gaussian_cloud_params(self):
        """
        Returns
        -------
        ndarray of UFloat
            (x, y, width, height, rotation, amplitude, offset)
            upopt: parameters of the fit with uncertainties.
            Positions (x, y) and widths (width, height) are in meters;
            rotation is in radians,
            (amplitude, offset) are given in counts.
        """
        popt, pcov = self.image.roi_fit_gaussian2d(self.atoms_roi)
        upopt = uncertainties.correlated_values(popt, pcov)

        pixel_size = self.analysis_config.imaging_system.atom_plane_pixel_size
        return np.asarray(upopt) * (pixel_size, pixel_size, pixel_size, pixel_size, 1, 1, 1)

    def process_shot(self, cloud_fit=None) -> str:
        """
        Process a single shot of bulk gas.

        Parameters
        ----------
        cloud_fit : {'gaussian', None}, default=None
            If 'gaussian', fit the atom cloud to a 2D Gaussian and store the parameters.
            If None, do not fit the atom cloud. No parameters are stored.

        Returns
        -------
        str
            Path to the processed results file
        """
        if cloud_fit == 'gaussian':
            gauss_params = self.get_gaussian_cloud_params()
            gauss_nom = unumpy.nominal_values(gauss_params)
            gauss_cov = uncertainties.covariance_matrix(gauss_params)

        atom_number = self.get_atom_number(method='sum', subtraction='double')
        run_number = self.run_number
        fname = Path(self.folder_path) / 'bulkgas_preprocess.h5'
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.attrs['n_runs'] = self.n_runs
                f.create_dataset('atom_numbers', data=[atom_number], maxshape=(self.n_runs,))
                if cloud_fit == 'gaussian':
                    n_params = 7
                    f.create_dataset('gaussian_cloud_params_nom', data=[gauss_nom], maxshape=(self.n_runs, n_params))
                    f['gaussian_cloud_params_nom'].attrs['fields'] = ['x', 'y', 'width', 'height', 'amplitude', 'offset']
                    f['gaussian_cloud_params_nom'].attrs['units'] = ['m', 'm', 'm', 'm', 'rad', 'counts', 'counts']
                    f.create_dataset('gaussian_cloud_params_cov', data=[gauss_cov], maxshape=(self.n_runs, n_params, n_params))

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
                f['atom_numbers'].resize(run_number + 1, axis=0)
                f['atom_numbers'][run_number] = atom_number

                # save parameters from runmanager globals
                f['current_params'].resize(run_number + 1, axis=0)
                f['current_params'][run_number] = self.current_params

                if cloud_fit == 'gaussian':
                    f['gaussian_cloud_params_nom'].resize(run_number + 1, axis=0)
                    f['gaussian_cloud_params_nom'][run_number] = gauss_nom
                    f['gaussian_cloud_params_cov'].resize(run_number + 1, axis=0)
                    f['gaussian_cloud_params_cov'][run_number] = gauss_cov

        return fname

    def show_images(self, fig: Optional[Figure] = None, raw_img_scale = 100) -> None:
        """
        Show the images of raw image/background image full frame and background subtracted images in ROIs
        We use the convention show_images() to show camera images at the processing level
        """
        if fig is None:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(10, 10),
                layout='constrained',
            )
        else:
            axs = fig.subplots(nrows=2, ncols=2)

        fig.suptitle(
            self.h5_path,
            fontsize=8,
        )
        fig.supxlabel('Length (mm)')
        fig.supylabel('Length (mm)')
        plot_unit = 1e-3
        plot_units_per_pixel = self.imaging_setup.atom_plane_pixel_size / plot_unit

        img_obj = self.image

        axs[0, 0].set_title('Raw, with atoms')
        axs[0, 1].set_title('Raw, without atoms')
        for i, img in enumerate([img_obj.array, img_obj.background]):
            cast(Axes, axs[0, i]).imshow(
                img,
                cmap='magma',
                vmax=raw_img_scale,
                extent=plot_units_per_pixel * np.array([0, img.shape[1], img.shape[0], 0]),
            )

        atoms_roi = self.atoms_roi
        axs[0, 0].add_patch(atoms_roi.patch(scale_factor=plot_units_per_pixel))
        axs[1, 0].set_title('Cloud Region \n(background subtracted)')

        img_obj.imshow_view(
            atoms_roi,
            scale_factor=plot_units_per_pixel,
            ax=axs[1, 0],
            cmap='magma',
            vmin=0,
        )

        bkg_roi = self.background_roi
        axs[0, 1].add_patch(bkg_roi.patch(scale_factor=plot_units_per_pixel))
        axs[1, 1].set_title(
            f'Background region \n(background subtracted)\nmean {img_obj.roi_mean(bkg_roi):.2f}, stddev {img_obj.roi_stddev(bkg_roi):.2f}',
        )

        img_obj.imshow_view(
            bkg_roi,
            scale_factor=plot_units_per_pixel,
            ax=axs[1,1],
            cmap='coolwarm',
            vmin=-20,
            vmax=+20,
        )
