from dataclasses import dataclass
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import os
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    lyse
except:
    import lyse
from analysis.data import h5lyze as hz
# from analysis.data import autolyze as az
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

@dataclass
class ROIConfig:
    """Configuration for ROIs in tweezer analysis.

    Parameters
    ----------
    roi_x : List[int]
        X-coordinates [start, end] for atom imaging region
        Only required when load_roi=False
    site_roi : dict,
        Dictionary containing site ROI coordinates
        Only required when load_roi=False
        Must have 'site_roi_x' and 'site_roi_y' arrays
    """
    def __init__(self, roi_x: List[int] = None, site_roi: dict = None):
        self.roi_x = roi_x
        self.site_roi = site_roi

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ROIConfig":
        """Load ROI configuration from YAML file.
        
        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file
            
        Returns
        -------
        ROIConfig
            ROI configuration object
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract ROI coordinates if present
        roi_x = config.get('roi_x', None)
        site_roi = config.get('site_roi', None)

        return cls(roi_x=roi_x, site_roi=site_roi)


@dataclass
class ImagingCamera:
    pixel_size: float
    """Pixel size, in m"""

    image_size: float
    """Size of sensor, in pixels (assumes square sensor)"""

    quantum_efficiency: float
    """Quantum efficiency of the camera"""

    gain: float
    """
    Additional conversion from detected electron counts to camera counts,
    (for cameras like EMCCDs).
    """

    image_name: str
    """
    Name of the image file to load.
    """


@dataclass
class ImagingSetup:
    imaging_f: float
    """Focal length of imaging lens, in m"""

    objective_f: float
    """Focal length of objective lens, in m"""

    lens_diameter: float  # diameter of light collection lenses
    """Diameter of imaging and objective lenses (or the smaller of the two), in m"""

    imaging_loss: float
    """Fractional power loss along imaging path."""

    camera: ImagingCamera
    """Detection camera"""

    @property
    def magnification(self):
        return self.imaging_f / self.objective_f

    @property
    def pixel_size_before_maginification(self):
        return  self.camera.pixel_size/self.magnification

    @property
    def solid_angle_fraction(self):
        """
        Solid angle captured by the imaging setup as a fraction of 4\pi.
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
        scattering_rate: float
            Scattering rate (\Gamma) for atoms being imaged.
        exposure_time: float
            Exposure time of the image in seconds.

        Returns
        -------
        float
            Camera counts per atom for this imaging setup.
        """
        count_rate_per_atom = (
            scattering_rate/2
            * self.solid_angle_fraction
            * self.imaging_loss
            * self.camera.quantum_efficiency
            * self.camera.gain
        )
        return count_rate_per_atom * exposure_time


manta_camera = ImagingCamera(
    pixel_size=5.5e-6,
    image_size=2048,
    quantum_efficiency=0.4,
    gain=1,
    image_name='manta419b_mot_images'
)


manta_setup = ImagingSetup(
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
    image_name='kinetix_images'
)

kinetix_setup = ImagingSetup(
    imaging_f=40.4e-3,
    objective_f=300e-3,
    lens_diameter=50.8e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=kinetix_camera,
)