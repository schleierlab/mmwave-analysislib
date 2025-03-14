import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

try:
    import lyse
except ImportError:
    from analysis.data import h5lyze as lyse

from .image_preprocessor import ImagePreProcessor
from .analysis_config import TweezerAnalysisConfig, ImagingSystem

class TweezerAnalysis(ImagePreProcessor):
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
        # Standard file locations
        self.multishot_path = 'X:\\userlib\\analysislib\\scripts\\multishot'

        # Initialize parent class first
        super().__init__(
            imaging_setup=config.imaging_system,
            load_type=load_type,
            h5_path=h5_path
        )

        # Store config
        self.config = config

        # Load ROIs and set class attributes
        self.atom_roi, self.background_roi, self.site_roi = self.load_roi()
            
        # Set threshold
        self.threshold = self.load_threshold()

        # Process images
        self.atom_images, self.background_images, self.sub_images = self.get_image_bkg_sub()
        self.roi_atoms, self.roi_bkgs = self.get_images_roi()

    def load_roi(self):
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

        if self.config.load_roi:
            # Load ROIs from standard .npy files
            roi_paths = {
                'site_roi_x': os.path.join(self.multishot_path, "site_roi_x.npy"),
                'site_roi_y': os.path.join(self.multishot_path, "site_roi_y.npy"),
                'roi_x': os.path.join(self.multishot_path, "roi_x.npy")
            }
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
        return [roi_x, roi_y], [self.config.bkg_roi_x, roi_y], [site_roi_x, site_roi_y]    

    def load_threshold(self):
        """Load threshold value from file or use default."""
        default_threshold = self.config.threshold

        if self.config.load_threshold:
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
        folder_path = self.folder_path

        # Set up paths for background images
        alternative_bkg_path = os.path.join(folder_path, 'alternative_bkg')
        average_bkg_path = os.path.join(folder_path, 'avg_shot_bkg.npy')
        last_bkg_sub_path = os.path.join(folder_path, 'last_bkg_sub')

        if self.config.method == 'alternative':
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
        elif self.config.method == 'average':
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
        """Get the summed counts in each site ROI.

        Parameters
        ----------
        sub_image : ndarray
            Background-subtracted image to analyze

        Returns
        -------
        site_roi_sums : list
            List of summed counts in each site ROI
        """
        site_roi_sums = []
        site_roi_x = self.site_roi[0]
        site_roi_y = self.site_roi[1]

        for i in np.arange(site_roi_x.shape[0]):
            y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
            x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]

            site_roi_signal = sub_image[y_start:y_end, x_start:x_end]
            signal_sum = np.sum(site_roi_signal)
            site_roi_sums.append(signal_sum)

        return site_roi_sums

    def analyze_site_existence(self, site_roi_sums):
        """Analyze whether atoms exist in each site based on the summed counts.

        Parameters
        ----------
        site_roi_sums : list
            List of summed counts in each site ROI
        """
        atom_exist_lst = []

        for signal_sum in site_roi_sums:
            if signal_sum > self.threshold:
                atom_exist_lst.append(1)
            else:
                atom_exist_lst.append(0)

        return atom_exist_lst

    def site_existence_lst(self):
        """
        Analyze whether atoms exist in each site for all images in one shot.
        
        Returns
        -------
        atom_exist_lst : list
            List of lists, where each inner list contains 1 or 0 for each site in each image
        """
        num_of_imgs = len(self.sub_images)
        
        # Get site counts and analyze existence for each image
        atom_exist_lst = []
        for i in range(num_of_imgs):
            sub_image = self.sub_images[i]
            roi_number_lst_i = self.get_site_counts(sub_image)
            atom_exist_lst_i = self.analyze_site_existence(roi_number_lst_i)
            atom_exist_lst.append(atom_exist_lst_i)

        return atom_exist_lst

    def calculate_survival_rate(self, method='default') -> tuple[float, float]:
            """Calculate survival rate (and uncertainty thereof) from atom existence lists.

            Parameters
            ----------
            atom_exist_lst_1, atom_exist_lst_2 : array_like, shape (n_sites,)
                Lists of atom existence values for the two tweezer images.
            method : {'default', 'laplace'}, optional
                Method for calculating the survival rate and uncertainty.
                Defaults to 'default' (which does the usual thing)

                'laplace': estimate the survival rate (and associated uncertainty)
                using Laplace's rule of succession (https://en.wikipedia.org/wiki/Rule_of_succession),
                whereby we inflate the number of atoms by 2 and the number of survivors by 1.
                Seems to be a better metric for closed-loop optimization by M-LOOP.
            """
            atom_exist_lst = self.site_existence_lst()
            n_initial_atoms = np.sum(atom_exist_lst[0])
            survivors = sum(1 for x,y in zip(atom_exist_lst[0], atom_exist_lst[1])
                    if x == 1 and y == 1)

            if method == 'default':
                survival_rate = survivors / n_initial_atoms
                uncertainty = np.sqrt(survival_rate * (1 - survival_rate) / n_initial_atoms)
            elif method == 'laplace':
                # expectation value of posterior beta distribution
                survival_rate = (survivors + 1) / (n_initial_atoms + 2)

                # calculate based on binomial distribution
                # uncertainty = np.sqrt(survival_rate * (1 - survival_rate) / (n_initial_atoms + 2))

                # sqrt of variance of the posterior beta distribution
                uncertainty = np.sqrt((survivors + 1) * (n_initial_atoms - survivors + 1) / ((n_initial_atoms + 3) * (n_initial_atoms + 2) ** 2))
            return survival_rate, uncertainty

    def save_data(self, folder_path, survival_rate, loop_var, site_counts_lst, run_number):
        """Save analysis results to files."""
        count_file_path = os.path.join(folder_path, 'tweezer_data.csv')
        site_counts_lst_file_path = os.path.join(folder_path, 'site_counts_lst.npy')

        if run_number == 1:  # Initialize files for first run
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            np.save(site_counts_lst_file_path, site_counts_lst)
        else:  # Append to existing files
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            roi_number_lst_old = np.load(site_counts_lst_file_path)
            roi_number_lst_new = np.dstack((roi_number_lst_old, site_counts_lst))
            np.save(site_counts_lst_file_path, roi_number_lst_new)

    
