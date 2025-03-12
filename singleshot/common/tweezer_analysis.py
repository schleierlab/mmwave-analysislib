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
from image_preprocessor import ImagePreProcessor
from .image_config import AnalysisConfig, ImagingSystem

class TweezerAnalysis(ImagePreProcessor):
    """Analysis class for tweezer imaging data.
    
    This class provides functionality for analyzing tweezer imaging data, including
    ROI-based analysis, background subtraction, and threshold-based detection.
    
    The class uses a configuration-based approach where all analysis parameters
    are specified through an AnalysisConfig object, which includes imaging system
    setup, ROI definitions, and analysis parameters.
    """

    def __init__(
            self,
            config: AnalysisConfig,
            load_type: str = 'lyse',
            h5_path: str = None):
        """Initialize TweezerAnalysis with analysis configuration.

        Parameters
        ----------
        config : AnalysisConfig
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
        # Standard file locations
        self.multishot_path = 'X:\\userlib\\analysislib\\scripts\\multishot'

        # Initialize parent class first
        super().__init__(
            imaging_setup=config.imaging_system,
            load_type=load_type,
            h5_path=h5_path
        )

        # Store config
        self.analysis_config = config

        # Load ROIs and set class attributes
        self.atom_roi, self.background_roi, self.site_roi = self.load_roi(
            roi_config_path=config.roi_config_path,
            bkg_roi_x=config.bkg_roi_x,
            load_roi=config.load_roi
        )
            
        # Set threshold
        self.threshold = self.load_threshold(config.load_threshold, config.threshold)

        # Process images
        self.atom_images, self.background_images, self.sub_images = self.get_image_bkg_sub(method=config.method)
        self.roi_atoms, self.roi_bkgs = self.get_images_roi()

    def load_roi(self, roi_config_path: str, bkg_roi_x: List[int], load_roi: bool):
        """Load ROI and site ROI either from file or from provided configuration.
        
        Parameters
        ----------
        roi_config_path : str
            Path to YAML configuration file containing ROI specifications.
            When load_roi=False:
            - Must contain 'roi_x' key with [start, end] coordinates
            - Must contain 'site_roi' section with 'site_roi_x' and 'site_roi_y'
        bkg_roi_x : List[int]
            X-coordinates [start, end] for background region.
            This is passed directly from __init__ rather than loaded from YAML.
        load_roi : bool
            If True, load ROIs from standard file locations
            If False, use ROIs from YAML config
            
        Returns
        -------
        atom_roi : List[List[int]]
            [[x_min, x_max], [y_min, y_max]] for atom ROI
        background_roi : List[List[int]]
            [[x_min, x_max], [y_min, y_max]] for background ROI
        site_roi : List[List[List[int]]]
            [site_roi_x, site_roi_y] for site-specific ROIs
        """
        # Get y-coordinates from globals
        try:
            roi_y = self.globals["tw_kinetix_roi_row"]
            print(f"Using tw_kinetix_roi_row from globals: {roi_y}")
        except KeyError:
            print("Warning: tw_kinetix_roi_row not found in globals, using full camera height")
            roi_y = [0, self.imaging_setup.camera.image_size]

        # Standard file locations for ROIs
        roi_paths = {
            'site_roi_x': os.path.join(self.multishot_path, "site_roi_x.npy"),
            'site_roi_y': os.path.join(self.multishot_path, "site_roi_y.npy"),
            'roi_x': os.path.join(self.multishot_path, "roi_x.npy")
        }

        if load_roi:
            # Load ROIs from standard .npy files
            try:
                roi_x = np.load(roi_paths['roi_x']).tolist()
                site_roi_x = np.load(roi_paths['site_roi_x'])  # Keep as numpy array for calculations
                site_roi_y = np.load(roi_paths['site_roi_y'])
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load ROI files from {self.multishot_path}: {e}")

            # Adjust site_roi_x relative to roi_x
            site_roi_x = site_roi_x - roi_x[0]

            # Add bkg site roi
            site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], np.array(site_roi_x)])
            site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) - 10], np.array(site_roi_y)])

        else:
            # When not loading from files, validate YAML config
            config = AnalysisConfig.from_yaml(roi_config_path)
            if config.roi_x is None:
                raise ValueError("When load_roi is False, the YAML config must include 'roi_x' coordinates")
            if config.site_roi is None:
                raise ValueError("When load_roi is False, the YAML config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

            # Load ROIs from YAML
            roi_x = config.roi_x
            site_roi_x = np.array(config.site_roi['site_roi_x'])
            site_roi_y = np.array(config.site_roi['site_roi_y'])

        # Always get roi_y from globals
        roi_y = self.globals["tw_kinetix_roi_row"]

        # Return ROIs in the format expected by the class
        return [roi_x, roi_y], [bkg_roi_x, roi_y], [site_roi_x, site_roi_y]    

    def load_threshold(self, load_threshold, default_threshold):
        """Load threshold value from file or use default."""
        if load_threshold:
            threshold_path = os.path.join(self.multishot_path, "th.npy")
            try:
                threshold = np.load(threshold_path)[0]
                print(f'Loaded threshold {threshold:.2f} from {threshold_path}')
                return threshold
            except FileNotFoundError:
                print(f'Warning: Threshold file not found at {threshold_path}, using default value')
                threshold = default_threshold
        else:
            if default_threshold is None:
                raise ValueError(
                    'When load_threshold is False, default_threshold must be provided'
                )
            return default_threshold

    def get_image_bkg_sub(self, method: str = 'average'):
        """Get background-subtracted images.

        Parameters
        ----------
        method : str, default='average'
            Method for background subtraction:
            - 'average': Use average background subtraction
            - 'alternative': Use alternative background subtraction

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
        folder_path = self.folder_path

        # Set up paths for background images
        alternative_bkg_path = os.path.join(folder_path, 'alternative_bkg')
        average_bkg_path = os.path.join(folder_path, 'avg_shot_bkg.npy')
        last_bkg_sub_path = os.path.join(folder_path, 'last_bkg_sub')

        if method == 'alternative':
            if self.globals['mot_do_coil']:
                atom_images = images
                background_images = np.load(alternative_bkg_path)
                sub_images = atom_images - background_images
                np.save(last_bkg_sub_path, sub_images)
            else:
                background_images = images
                np.save(alternative_bkg_path, images)
                sub_images = np.load(last_bkg_sub_path)
                # load last background subtracted images
                # during background taking shot to make
                # sure there is something to plot
        elif method == 'average':
            atom_images = images
            background_images = np.load(average_bkg_path)
            sub_images = atom_images - background_images
            np.save(last_bkg_sub_path, sub_images)
        else:
            raise NotImplementedError
            #TODO implement the method here

        return atom_images, background_images, sub_images
    
    def convert_dict_to_array(self, dict):
        """
        convert the images from dictionary to np.array
        with shape(# shots in a single sequence, size of images)
        """
        dict_types = list(dict.keys())
        array = []
        for item in dict_types:
            array.append(dict[item])
        return np.array(array)

    def get_images_roi(self):
        """Get ROI images for atoms and background regions.

        Returns
        -------
        roi_atoms : array
            Images cropped to atoms ROI region
        roi_bkgs : array
            Images cropped to background ROI region
        """
        [roi_x, _] = self.atom_roi
        [roi_x_bkg, _] = self.background_roi
        sub_images = self.sub_images
        roi_atoms = sub_images[:, :, roi_x[0]:roi_x[1]]
        roi_bkgs = sub_images[:, :, roi_x_bkg[0]:roi_x_bkg[1]]
        return roi_atoms, roi_bkgs
    
    def get_site_counts(self, sub_image):
        """Get counts in each tweezer site ROI.

        Returns
        -------
        roi_number_lst : list
            List of summed counts in each site ROI
        """
        roi_number_lst = []
        site_roi_x = self.site_roi[0]
        site_roi_y = self.site_roi[1]

        for i in np.arange(site_roi_x.shape[0]):
            y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
            x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]

            site_roi_signal = sub_image[y_start:y_end, x_start:x_end]
            signal_sum = np.sum(site_roi_signal)
            roi_number_lst.append(signal_sum)

        return roi_number_lst

    def analyze_site_existence(self, roi_number_lst):
        """Analyze atom existence in each site and create visualization rectangles.

        Parameters
        ----------
        roi_number_lst : list
            List of summed counts in each site ROI from get_site_counts

        Returns
        -------
        rect_sig : list
            List of Rectangle patches for visualization
        atom_exist_lst : list
            List of booleans indicating atom presence in each site
        """
        rect_sig = []
        atom_exist_lst = []
        site_roi_x = self.site_roi[0]
        site_roi_y = self.site_roi[1]

        for i, signal_sum in enumerate(roi_number_lst):
            y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
            x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]

            if signal_sum > self.threshold:
                rect = patches.Rectangle(
                    (x_start, y_start),
                    x_end - x_start,
                    y_end - y_start,
                    linewidth=1,
                    edgecolor='gold',
                    facecolor='none',
                    alpha=0.5,
                    fill=False)
                rect_sig.append(rect)
                atom_exist_lst.append(1)
            else:
                atom_exist_lst.append(0)

        return rect_sig, atom_exist_lst