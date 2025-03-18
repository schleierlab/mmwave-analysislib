from dataclasses import dataclass
from typing import Literal, Optional

import yaml

from .image import ROI


@dataclass
class ImagingCamera:
    """Configuration class for imaging camera parameters.

    Parameters
    ----------
    pixel_size : float
        Physical size of camera pixels in meters
    image_size : float
        Size of camera sensor in pixels
    quantum_efficiency : float
        Quantum efficiency of camera
    gain : float
        Camera gain setting
    image_group_name : str
        Name of the group in the h5 file containing the camera images
    image_name_stem : str
        Stem of the names for the datasets holding the camera images;
        full names will be
            f'{image_name_stem}{shot_number}'
        such as 'kinetix0', ...
    """
    pixel_size: float
    image_size: float
    quantum_efficiency: float
    gain: float
    image_group_name: str
    image_name_stem: str


@dataclass
class ImagingSystem:
    """Configuration class for imaging system parameters.

    Parameters
    ----------
    imaging_f : float
        Focal length of imaging lens in meters
    objective_f : float
        Focal length of objective lens in meters
    lens_diameter : float
        Diameter of imaging lens in meters
    imaging_loss : float
        Loss factor in imaging system
    camera : ImagingCamera
        Camera configuration object
    """
    imaging_f: float
    objective_f: float
    lens_diameter: float
    imaging_loss: float
    camera: ImagingCamera

    def magnification(self):
        """Calculate imaging system magnification."""
        return self.imaging_f / self.objective_f

    @property
    def atom_plane_pixel_size(self):
        """Get effective pixel size before magnification."""
        return self.camera.pixel_size / self.magnification()

    @property
    def solid_angle_fraction(self):
        """Calculate solid angle fraction captured by imaging system.

        Makes a small angle approximation for the imaging NA.
        """
        return (self.lens_diameter / 2)**2 / (4 * self.imaging_f**2)

    def counts_per_atom(self, scattering_rate, exposure_time):
        """
        For an image of atoms with given scattering rate taken with a given exposure time,
        find the camera counts per atom we expect with this imaging setup.
        Note that the scattering rate is for resonance light not for the detuned light.

        Parameters
        ----------
        scattering_rate : float
            Scattering rate (Γ) for atoms being imaged
        exposure_time : float
            Exposure time in seconds

        Returns
        -------
        float
            Expected camera counts per atom
        """
        count_rate_per_atom = (
            scattering_rate/2
            * self.solid_angle_fraction
            * self.imaging_loss
            * self.camera.quantum_efficiency
            * self.camera.gain
        )
        return count_rate_per_atom * exposure_time


# Pre-configured camera systems
manta_camera = ImagingCamera(
    pixel_size=5.5e-6,
    image_size=2048,
    quantum_efficiency=0.4,
    gain=1,
    image_group_name='manta419b_mot_images',
    image_name_stem='manta419b_mot',
)

manta_system = ImagingSystem(
    imaging_f=50e-3,
    objective_f=125e-3,
    lens_diameter=25.4e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=manta_camera,
)

kinetix_camera = ImagingCamera(
    pixel_size=6.5e-6,
    image_size=2400,
    quantum_efficiency=0.58,
    gain=1,
    image_group_name='kinetix_images',
    image_name_stem='kinetix',
)

kinetix_system = ImagingSystem(
    imaging_f=40.4e-3,
    objective_f=300e-3,
    lens_diameter=50.8e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=kinetix_camera,
)


@dataclass
class BulkGasAnalysisConfig:
    """Configuration class for bulk gas analysis parameters.

    Parameters
    ----------
    imaging_system : ImagingSystem
        Configuration object for the imaging system setup
    exposure_time : float
        Imaging exposure time in seconds
    atoms_roi : list[list[int]]
        ROI for atoms in bulk gas analysis, in format:
        [[x_min, x_max], [y_min, y_max]]
    bkg_roi : list[list[int]]
        ROI for background in bulk gas analysis, in format:
        [[x_min, x_max], [y_min, y_max]]
    """
    imaging_system: ImagingSystem
    exposure_time: float
    atoms_roi: ROI
    bkg_roi: ROI


@dataclass
class TweezerAnalysisConfig:
    """Configuration class for image analysis parameters.

    This class consolidates all configuration parameters needed for analyzing
    imaging data, including the imaging system setup, ROI definitions, and
    analysis-specific parameters.

    Parameters
    ----------
    imaging_system : ImagingSystem
        Configuration object for the imaging system setup
    method : str, default='average'
        Method for background subtraction:
        - 'average': Use average background subtraction
        - 'alternating': Use alternating background subtraction
    bkg_roi_x : Tuple[int, int]
        X-coordinates [start, end] for background region
    load_roi : bool, default=True
        If True, load ROIs from standard .npy files:
        - roi_x.npy: Main ROI x-coordinates
        - site_roi_x.npy: Site ROI x-coordinates
        - site_roi_y.npy: Site ROI y-coordinates
        If False, load from YAML config
    roi_config_path : Optional[str], default=None
        Path to YAML configuration file for ROIs. Required when load_roi=False
    roi_x : Optional[list[int]], default=None
        X-coordinates [start, end] for atom imaging region.
        Only required when load_roi=False
    site_roi : Optional[Dict[str, list[list[int]]]], default=None
        Dictionary containing site ROI coordinates. Only required when load_roi=False.
        Must have:
        - 'site_roi_x': List of [start, end] x-coordinates for each site
        - 'site_roi_y': List of [start, end] y-coordinates for each site
    load_threshold : bool, default=True
        Whether to load threshold from file for tweezer analysis
    threshold : Optional[float], default=None
        Threshold value to use if not loading from file
    """
    imaging_system: ImagingSystem = kinetix_system
    method: Literal['average', 'alternating'] = 'average'
    bkg_roi_x: tuple[int, int] = (1900, 2400)
    load_roi: bool = True
    roi_config_path: Optional[str] = None
    roi_x: Optional[tuple[int, int]] = None
    site_roi: Optional[dict[str, ROI]] = None
    load_threshold: bool = True
    threshold: Optional[float] = None

    # TODO: do we actually want to store defaults in a YAML file?
    # If so, we can store here: 'X:\\userlib\\analysislib\\scripts\\multishot\\tweezer_roi.yaml'
    # Or do we just want to hard code the in the class?
    # YAML makes sense if we are in fact changing the defaults often and have a huge list of them
    # If we don't have one of those conditions then we might as well just hard code it

    @classmethod
    def from_yaml(cls, path: str): # not being used for now
        """Create an AnalysisConfig instance from a YAML file.

        Parameters
        ----------
        path : str
            Path to YAML configuration file

        Returns
        -------
        AnalysisConfig
            Analysis configuration object
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
