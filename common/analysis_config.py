from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
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
    '''Focal length of the imaging lens (that produces the camera image), in meters'''

    objective_f: float
    '''Focal length of the objective lens, in meters'''

    lens_diameter: float
    '''Diameter of the objective lens, in meters'''

    imaging_loss: float
    '''Power loss factor in the imaging system (should be between 0 and 1)'''

    camera: ImagingCamera
    '''Camera in the imaging system.'''

    def magnification(self):
        """Calculate imaging system magnification."""
        return self.imaging_f / self.objective_f

    @property
    def atom_plane_pixel_size(self):
        """Get effective pixel size before magnification."""
        return self.camera.pixel_size / self.magnification()

    @property
    def solid_angle_fraction(self):
        """
        Calculate solid angle fraction captured by imaging system.
        """
        cone_half_angle = np.arctan(self.lens_diameter / (2 * self.objective_f))
        return (1 - np.cos(cone_half_angle)) / 2

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
    image_name_stem='manta',
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
    imaging_f=300e-3,
    objective_f=40.4e-3,
    lens_diameter=50.8e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=kinetix_camera,
)

# TODO: do we need this? We were able to eliminate TweezerAnalysisConfig
@dataclass
class BulkGasAnalysisConfig:
    """Configuration class for bulk gas analysis parameters.

    Parameters
    ----------
    imaging_system : ImagingSystem
        Configuration object for the imaging system setup
    exposure_time : float
        Imaging exposure time in seconds
    atoms_roi, bkg_roi : ROI
        ROIs for atoms and background in bulk gas analysis
    """
    imaging_system: ImagingSystem
    exposure_time: float
    atoms_roi: ROI
    bkg_roi: ROI
