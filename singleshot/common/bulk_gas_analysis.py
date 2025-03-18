from pathlib import Path
from typing import ClassVar, Literal

import h5py
from scipy.constants import pi, k as k_B

try:
    lyse
except:
    import lyse

from analysislib.analysis.data import h5lyze as hz
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

    m_cesium = 2.20694650e-25
    """ Cesium mass in kg"""

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

        images = self.images
        image_types = list(images.keys())
        atom_image = images[image_types[0]]
        background_image = images[image_types[1]]
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

    # def save_atom_number(self, atom_number):
    #     """Save atom number to the HDF5 file."""
    #     self.save_processed_quantities(atom_number=atom_number)

    # def save_atom_temperature(self, atom_number, time_of_flight, gaussian_waist, temperature):
    #     """Save temperature measurement data to the HDF5 file."""
    #     self.save_processed_quantities(
    #         atom_number=atom_number,
    #         time_of_flight=time_of_flight,
    #         gaussian_waist=gaussian_waist,
    #         temperature=temperature
    #     )

    # def load_atom_number(self):
    #     """Load atom number from the HDF5 file.

    #     Returns
    #     -------
    #     float
    #         Atom number for this run
    #     """
    #     try:
    #         result = self.load_processed_quantities('atom_number')
    #         return result['atom_number']
    #     except (FileNotFoundError, ValueError) as e:
    #         print(f"Warning: Could not load atom number: {e}")
    #         return None

    # def load_atom_temperature(self):
    #     """Load temperature measurement data from the HDF5 file.

    #     Returns
    #     -------
    #     tuple
    #         (time_of_flight, waist_x, waist_y, temperature_x, temperature_y)
    #     """
    #     try:
    #         result = self.load_processed_quantities(
    #             'time_of_flight',
    #             'gaussian_waist',
    #             'temperature'
    #         )

    #         time_of_flight = result['time_of_flight']
    #         gaussian_waist = result['gaussian_waist']
    #         temperature = result['temperature']

    #         # Split waist and temperature into x,y components
    #         waist_x = gaussian_waist[0]
    #         waist_y = gaussian_waist[1]
    #         temperature_x = temperature[0]
    #         temperature_y = temperature[1]

    #         return time_of_flight, waist_x, waist_y, temperature_x, temperature_y
    #     except (FileNotFoundError, ValueError) as e:
    #         print(f"Warning: Could not load temperature data: {e}")
    #         return None, None, None, None, None

    # @classmethod
    # def get_atom_temperature(cls, time_of_flight, gaussian_waist):
    #     """Get atom temperature from cloud size at different time of flight.

    #     The cloud size is fitted with a Gaussian function to extract the waist.
    #     The temperature is calculated from the waist size vs time of flight.

    #     Parameters
    #     ----------
    #     time_of_flight : float
    #         Time of flight in seconds
    #     gaussian_waist : tuple
    #         (x_waist, y_waist) in meters

    #     Returns
    #     -------
    #     temperature : tuple
    #         (x_temperature, y_temperature) in Kelvin
    #     """
    #     return cls.m_cesium / k_B * (gaussian_waist / time_of_flight)**2

    def process_shot(self):
        atom_number = self.get_atom_number(method='sum', subtraction='double')

        run_number = self.run_number
        fname = Path(self.folder_path) / 'bulkgas_preprocess.h5'
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.create_dataset('atom_numbers', data=[atom_number], maxshape=(self.n_runs,))
        else:
            with h5py.File(fname, 'a') as f:
                f['atom_numbers'].resize(run_number + 1, axis=0)
                f['atom_numbers'][run_number] = atom_number

        return fname
