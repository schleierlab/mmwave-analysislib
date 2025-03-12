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
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

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
class ImagingSystem:
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


@dataclass
class AnalysisConfig:
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
        - 'alternative': Use alternative background subtraction
    bkg_roi_x : List[int]
        X-coordinates [start, end] for background region
    load_roi : bool, default=True
        If True, load ROIs from standard .npy files:
        - roi_x.npy: Main ROI x-coordinates
        - site_roi_x.npy: Site ROI x-coordinates
        - site_roi_y.npy: Site ROI y-coordinates
        If False, load from YAML config
    roi_config_path : Optional[str], default=None
        Path to YAML configuration file for ROIs. Required when load_roi=False
    roi_x : Optional[List[int]], default=None
        X-coordinates [start, end] for atom imaging region.
        Only required when load_roi=False
    site_roi : Optional[Dict[str, List[List[int]]]], default=None
        Dictionary containing site ROI coordinates. Only required when load_roi=False.
        Must have:
        - 'site_roi_x': List of [start, end] x-coordinates for each site
        - 'site_roi_y': List of [start, end] y-coordinates for each site
    load_threshold : bool, default=True
        Whether to load threshold from file for tweezer analysis
    threshold : Optional[float], default=None
        Threshold value to use if not loading from file
    exposure_time : Optional[float], default=None
        Imaging exposure time in seconds, required for bulk gas analysis
    atoms_roi : Optional[List[List[int]]], default=None
        ROI for atoms in bulk gas analysis, in format:
        [[x_min, x_max], [y_min, y_max]]
    bkg_roi : Optional[List[List[int]]], default=None
        ROI for background in bulk gas analysis, in format:
        [[x_min, x_max], [y_min, y_max]]
    """
    imaging_system: ImagingSystem
    method: str = 'average'
    bkg_roi_x: List[int] = None
    load_roi: bool = True
    roi_config_path: Optional[str] = None
    roi_x: Optional[List[int]] = None
    site_roi: Optional[Dict[str, List[List[int]]]] = None
    load_threshold: bool = True
    threshold: Optional[float] = None
    exposure_time: Optional[float] = None
    atoms_roi: Optional[List[List[int]]] = None
    bkg_roi: Optional[List[List[int]]] = None
    # TODO: do we actually want to store defaults in a YAML file?
    # Or do we just want to hard code the in the class?
    # YAML makes sense if we are in fact changing the defaults often and have a huge list of them
    # If we don't have one of those conditions then we might as well just hard code it

    @classmethod
    def from_yaml(cls, path: str):
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
            config = yaml.safe_load(f)
        return cls(**config)

manta_camera = ImagingCamera(
    pixel_size=5.5e-6,
    image_size=2048,
    quantum_efficiency=0.4,
    gain=1,
    image_name='manta419b_mot_images'
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
    image_name='kinetix_images'
)

kinetix_system = ImagingSystem(
    imaging_f=40.4e-3,
    objective_f=300e-3,
    lens_diameter=50.8e-3,
    imaging_loss=1/1.028,  # from Thorlabs FBH850-10 line filter
    camera=kinetix_camera,
)