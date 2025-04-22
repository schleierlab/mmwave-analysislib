from pathlib import Path
from typing import ClassVar, Literal, cast, Optional

import h5py
import numpy as np
import uncertainties
import uncertainties.unumpy as unumpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.constants import pi

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
    images: tuple[Image, ...]

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

        background_exposure = self.exposures[-1]
        self.images = tuple(
            Image(atom_exposure, background_exposure)
            for atom_exposure in self.exposures[:-1]
        )

    def get_atom_numbers(
            self,
            method: Literal['sum', 'fit'] = 'sum',
            subtraction: Literal['simple', 'double'] = 'simple',
    ) -> NDArray:
        """
        sum of counts in roi/counts_per_atom - average background atom per px* roi size

        Parameters
        ----------
        method: {'sum', 'fit'}
            Method for atom number calculation. If 'sum', just take the total counts within a ROI
            and convert to an atom number. If 'fit', fit the observed images to a
            2D Gaussian and integrate the Gaussian to get the total counts.
        subtraction: {'simple', 'double'}
            Background subtraction method. If 'simple', do the naive thing.
            If 'double', use the background-subtracted images and further
            use a distant part of the images to estimate any further background
            in the already-background-subtracted image (which may arise from
            e.g. drifts in TA power.)

        Returns
        -------
        atom_number: NDArray, (n_images,)
        """
        if method == 'fit':
            raise NotImplementedError

        atom_counts = np.asarray([image.roi_sum(self.atoms_roi) for image in self.images]).astype(np.float_)
        if subtraction == 'double':
            area_ratio = self.atoms_roi.pixel_area / self.background_roi.pixel_area
            background_counts = np.asarray([image.roi_sum(self.background_roi) for image in self.images])
            atom_counts -= background_counts * area_ratio

        return atom_counts / self.counts_per_atom

    def get_gaussian_cloud_params(self, uniform = False):
        """
        Returns
        -------
        NDArray[UFloat], shape (n_images, number_of_parameters)
            A numpy array of (uncertain) Gaussian fit parameters for each image.
            The parameters, in order, are
                (x, y, width, height, rotation, amplitude, offset)
            Center coordinates (x, y) and widths (width, height) are in meters;
            rotation is in radians,
            (amplitude, offset) are given in counts.
            Note that when rotation = 0, we have only 6 parameters; if uniform, we have only 5
        """
        upopts = []
        for image in self.images:
            popt, pcov = image.roi_fit_gaussian2d(self.atoms_roi, uniform)
            upopt = uncertainties.correlated_values(popt, pcov)
            upopts.append(upopt)

        pixel_size = self.analysis_config.imaging_system.atom_plane_pixel_size
        if uniform:
            return np.asarray(upopts) * (pixel_size, pixel_size, pixel_size, 1, 1)
        else:
            return np.asarray(upopts) * (pixel_size, pixel_size, pixel_size, pixel_size, 1, 1)

    def process_shot(self, cloud_fit=None) -> str:
        """
        Process a single shot of bulk gas.

        Parameters
        ----------
        cloud_fit : {'gaussian', 'gaussian_uniform', None}, default=None
            If 'gaussian', fit the atom cloud to a 2D Gaussian and store the parameters.
            If 'gaussian_uniform', fit the atom cloud to a 2D Gaussian with uniform width and height.
            If None, do not fit the atom cloud. No parameters are stored.

        Returns
        -------
        str
            Path to the processed results file
        """
        if cloud_fit == 'gaussian':
            gauss_params = self.get_gaussian_cloud_params()
            gauss_nom = np.asarray([unumpy.nominal_values(params) for params in gauss_params])
            gauss_cov = np.asarray([uncertainties.covariance_matrix(params) for params in gauss_params])
            # integrated under 2D gaussian: 2 * pi * peak_height * sigma_u * sigma_v
            # need to express sigma_u, sigma_v in pixels
            gauss_atom_counts = 2 * pi * np.prod(gauss_params[:, [2, 3, 4]], axis=1) / (self.imaging_setup.atom_plane_pixel_size)**2
            gauss_atom_num = gauss_atom_counts / self.counts_per_atom
        elif cloud_fit == 'gaussian_uniform':
            gauss_params = self.get_gaussian_cloud_params(uniform=True)
            gauss_nom = np.asarray([unumpy.nominal_values(params) for params in gauss_params])
            gauss_cov = np.asarray([uncertainties.covariance_matrix(params) for params in gauss_params])
            # integrated under 2D gaussian: 2 * pi * peak_height * sigma^2
            # need to express sigma in pixels
            width = gauss_params[:, 2]  # Get the uniform width parameter
            amplitude = gauss_params[:, 3]  # Get the amplitude parameter
            gauss_atom_counts = 2 * np.pi * amplitude * width**2 / (self.imaging_setup.atom_plane_pixel_size)**2
            gauss_atom_num = gauss_atom_counts / self.counts_per_atom

        atom_numbers = self.get_atom_numbers(method='sum', subtraction='double')
        run_number = self.run_number
        fname = self.h5_path.with_name('bulkgas_preprocess.h5')
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.attrs['n_runs'] = self.n_runs
                f.create_dataset('atom_numbers', data=[atom_numbers], maxshape=(self.n_runs, len(self.images)))
                if cloud_fit == 'gaussian':
                    n_params = 6
                    f.create_dataset('gaussian_cloud_params_nom', data=[gauss_nom], maxshape=(self.n_runs, len(self.images), n_params))
                    f['gaussian_cloud_params_nom'].attrs['fields'] = ['x', 'y', 'width', 'height', 'amplitude', 'offset']
                    f['gaussian_cloud_params_nom'].attrs['units'] = ['m', 'm', 'm', 'm', 'counts', 'counts']
                    f.create_dataset('gaussian_cloud_params_cov', data=[gauss_cov], maxshape=(self.n_runs, len(self.images), n_params, n_params))

                    f.create_dataset(
                        'gaussian_atom_numbers_nom',
                        data=[unumpy.nominal_values(gauss_atom_num)],
                        maxshape=(self.n_runs, len(self.images)),
                    )
                    f.create_dataset(
                        'gaussian_atom_numbers_std',
                        data=[unumpy.std_devs(gauss_atom_num)],
                        maxshape=(self.n_runs, len(self.images)),
                    )
                elif cloud_fit == 'gaussian_uniform':
                    n_params = 5
                    f.create_dataset('gaussian_cloud_params_nom', data=[gauss_nom], maxshape=(self.n_runs, len(self.images), n_params))
                    f['gaussian_cloud_params_nom'].attrs['fields'] = ['x', 'y', 'width', 'amplitude', 'offset']
                    f['gaussian_cloud_params_nom'].attrs['units'] = ['m', 'm', 'm', 'counts', 'counts']
                    f.create_dataset('gaussian_cloud_params_cov', data=[gauss_cov], maxshape=(self.n_runs, len(self.images), n_params, n_params))

                    f.create_dataset(
                        'gaussian_atom_numbers_nom',
                        data=[unumpy.nominal_values(gauss_atom_num)],
                        maxshape=(self.n_runs, len(self.images)),
                    )
                    f.create_dataset(
                        'gaussian_atom_numbers_std',
                        data=[unumpy.std_devs(gauss_atom_num)],
                        maxshape=(self.n_runs, len(self.images)),
                    )

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
                f['atom_numbers'][run_number] = atom_numbers

                # save parameters from runmanager globals
                f['current_params'].resize(run_number + 1, axis=0)
                f['current_params'][run_number] = self.current_params

                if cloud_fit == 'gaussian' or cloud_fit == 'gaussian_uniform':
                    f['gaussian_cloud_params_nom'].resize(run_number + 1, axis=0)
                    f['gaussian_cloud_params_nom'][run_number] = gauss_nom
                    f['gaussian_cloud_params_cov'].resize(run_number + 1, axis=0)
                    f['gaussian_cloud_params_cov'][run_number] = gauss_cov
                    f['gaussian_atom_numbers_nom'].resize(run_number + 1, axis=0)
                    f['gaussian_atom_numbers_nom'][run_number] = unumpy.nominal_values(gauss_atom_num)
                    f['gaussian_atom_numbers_std'].resize(run_number + 1, axis=0)
                    f['gaussian_atom_numbers_std'][run_number] = unumpy.std_devs(gauss_atom_num)

        return fname

    def show_images(self, image_index: int = 0, fig: Optional[Figure] = None, raw_img_scale: int = 100) -> None:
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

        img_obj = self.images[image_index]

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

    def show_state_sensitive_images(self, fig: Optional[Figure] = None, ):
        fig.suptitle(self.h5_path, fontsize='x-small')

        if fig is None:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 10),
                layout='constrained',
            )
        else:
            axs = fig.subplots(nrows=2, ncols=1)

        plot_unit = 1e-3
        plot_units_per_pixel = self.imaging_setup.atom_plane_pixel_size / plot_unit

        for i, ax in enumerate(axs):
            im = self.images[i].imshow_view(
                self.atoms_roi,
                scale_factor=plot_units_per_pixel,
                ax=ax,
                cmap='magma',
                vmin=0,
                vmax=(20 if i == 0 else None),
            )
            fig.colorbar(im, ax=ax, location='right')
