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
from image_preprocessor import ImagePreProcessor

class TweezerAnalysis(ImagePreProcessor):

    def __init__(
            self,
            imaging_setup: ImagingSetup,
            method: str = 'average',
            bkg_roi_x: List[int] = [1900, 2400],
            load_roi: bool = True,
            roi_config_path: str = None,
            load_threshold: bool = True,
            threshold: Optional[float] = None,
            load_type: str = 'lyse',
            h5_path: str = None,):
        """Initialize TweezerAnalysis with ROI configuration and analysis parameters.

        Parameters
        ----------
        imaging_setup : ImagingSetup
            Imaging setup configuration object
        bkg_roi_x : List[int]
            X-coordinates [start, end] for background region
        roi_config_path : str
            Path to YAML configuration file for ROIs
        method : str, default='alternative'
            Method for background subtraction, one of:
            - 'alternative': Use alternative background subtraction
            - 'standard': Use standard background subtraction
        load_roi : bool, default=True
            If True, load roi_x and site ROIs from standard .npy files:
            - roi_x.npy: Main ROI x-coordinates
              - site_roi_x.npy: Site ROI x-coordinates
              - site_roi_y.npy: Site ROI y-coordinates
            If False, load from YAML config (requires roi_x and site_roi)
        load_type : str, optional
            Type of loading to perform
        h5_path : str, optional
            Path to h5 file to load
        load_threshold : bool, default=True
            Whether to load threshold from file
        threshold : float, optional
            Threshold value to use if not loading from file
        """
        # Standard file locations
        self.multishot_path = 'X:\\userlib\\analysislib\\scripts\\multishot'

        # Initialize parent class first
        super().__init__(
            imaging_setup=imaging_setup,
            load_type=load_type,
            h5_path=h5_path
        )

        # Load ROIs and set class attributes
        self.atom_roi, self.background_roi, self.site_roi = self.load_roi(
            roi_config_path=roi_config_path,
            bkg_roi_x=bkg_roi_x,
            load_roi=load_roi
        )
            
        # Set threshold
        self.threshold = self.load_threshold(load_threshold, threshold)

        # Process images
        self.atom_images, self.background_images, self.sub_images = self.get_image_bkg_sub(method=method)
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
            roi_config = ROIConfig.from_yaml(roi_config_path)
            # When not loading from files, validate YAML config
            if roi_config.roi_x is None:
                raise ValueError("When load_roi is False, the YAML config must include 'roi_x' coordinates")
            if roi_config.site_roi is None:
                raise ValueError("When load_roi is False, the YAML config must include a 'site_roi' section with 'site_roi_x' and 'site_roi_y' arrays")

            # Load ROIs from YAML
            roi_x = roi_config.roi_x
            site_roi_config = roi_config.site_roi
            site_roi_x = np.array(site_roi_config['site_roi_x'])
            site_roi_y = np.array(site_roi_config['site_roi_y'])

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

    def plot_images(self, roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True):
        """Plot the analysis results with optional site ROIs and background ROIs.
        
        Parameters
        ----------
        roi_image_scale : int, default=150
            Scale factor for ROI image colormaps
        show_site_roi : bool, default=True
            Whether to show site ROI rectangles
        plot_bkg_roi : bool, default=True
            Whether to plot background ROI images
        """
        num_of_imgs = len(self.sub_images)
        
        # Get site counts and analyze existence for each image
        rect_sig = []
        atom_exist_lst = []
        for i in range(num_of_imgs):
            sub_image = self.sub_images[i]
            roi_number_lst_i = self.get_site_counts(sub_image)
            rect_sig_i, atom_exist_lst_i = self.analyze_site_existence(roi_number_lst_i)
            rect_sig.append(rect_sig_i)
            atom_exist_lst.append(atom_exist_lst_i)

        # Create figure with enough rows for all shots plus background ROIs
        num_rows = num_of_imgs + (2 if plot_bkg_roi else 0)
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 5*num_rows), constrained_layout=True)
        
        # If only one subplot, wrap it in a list for consistent indexing
        if num_rows == 1:
            axs = [axs]

        plt.rcParams.update({'font.size': 14})
        fig.suptitle('Tweezer Array Imaging Analysis', fontsize=16)

        # Configure all axes
        for ax in axs:
            ax.set_xlabel('x [px]')
            ax.set_ylabel('y [px]')
            ax.tick_params(axis='both', which='major', labelsize=12)

        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

        # Plot tweezer ROIs for each shot
        for i in range(num_of_imgs):
            ax_tweezer = axs[i]
            ax_tweezer.set_title(f'Shot {i+1} Tweezer ROI', fontsize=14, pad=10)
            pos = ax_tweezer.imshow(self.roi_atoms[i], **roi_img_color_kw)
            
            if show_site_roi:
                # Draw the base ROI rectangles
                site_roi_x = self.site_roi[0]
                site_roi_y = self.site_roi[1]
                base_rects = []
                for j in range(site_roi_x.shape[0]):
                    y_start, y_end = site_roi_y[j,0], site_roi_y[j,1]
                    x_start, x_end = site_roi_x[j,0], site_roi_x[j,1]
                    rect = patches.Rectangle(
                        (x_start, y_start),
                        x_end - x_start,
                        y_end - y_start,
                        linewidth=1,
                        edgecolor='blue',
                        facecolor='none',
                        alpha=0.3)
                    base_rects.append(rect)
                pc_base = PatchCollection(base_rects, match_original=True)
                ax_tweezer.add_collection(pc_base)
                
                # Draw the signal rectangles
                if i < len(rect_sig):
                    pc_sig = PatchCollection(rect_sig[i], match_original=True)
                    ax_tweezer.add_collection(pc_sig)
            
            fig.colorbar(pos, ax=ax_tweezer).ax.tick_params(labelsize=12)

        # Plot background ROIs if requested
        if plot_bkg_roi:
            for i in range(min(2, num_of_imgs)):  # Plot up to 2 background ROIs
                ax_bkg = axs[num_of_imgs + i]
                ax_bkg.set_title(f'Shot {i+1} Background ROI', fontsize=14, pad=10)
                pos = ax_bkg.imshow(self.roi_bkgs[i], **roi_img_color_kw)
                fig.colorbar(pos, ax=ax_bkg).ax.tick_params(labelsize=12)

        # # Save the figure
        # fig.savefig(os.path.join(self.folder_path, 'atom_cloud.png'))
        # plt.close(fig)