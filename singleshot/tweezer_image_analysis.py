"""
Class-based implementation of tweezer image analysis functionality.
This module provides a structured approach to analyzing tweezer imaging data
with both average and alternating background subtraction methods.

Created on: 2024-12-26
"""

import os
from pathlib import Path

import h5py
import lyse
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from analysis.data import h5lyze as hz
from matplotlib.collections import PatchCollection


class TweezerImageAnalyzer:
    """
    A class to handle tweezer image analysis with different background subtraction methods.
    """

    def __init__(self, roi_config, roi_paths, show_site_roi=True, load_roi=True):
        """
        Initialize the TweezerImageAnalyzer with configuration parameters.

        Args:
            roi_config (dict): Configuration for ROI coordinates
            roi_paths (dict): Paths to ROI data files
            show_site_roi (bool): Flag to show site ROIs in plots
            load_roi (bool): Flag to load ROI data
        """
        self.roi_config = roi_config
        self.roi_paths = roi_paths
        self.show_site_roi = show_site_roi
        self.load_roi = load_roi
        self.multishot_path = str(Path(self.roi_paths['site_roi_x']).parent)

        # Initialize data attributes
        self.site_roi_x = None
        self.site_roi_y = None
        self.roi_x = None
        self.rect = None

        # Load ROI data if requested
        if load_roi:
            self._load_roi_data()

    def _load_roi_data(self):
        """Load ROI data from files."""
        self.site_roi_x = np.load(self.roi_paths['site_roi_x'])
        self.site_roi_y = np.load(self.roi_paths['site_roi_y'])
        self.roi_x = np.load(self.roi_paths['roi_x'])
        self.roi_config["x"][:] = self.roi_x

        # Add minimum value as first entry and adjust y-offset
        self.site_roi_x = np.concatenate([[np.min(self.site_roi_x, axis=0)], self.site_roi_x])
        self.site_roi_y = np.concatenate([[np.min(self.site_roi_y, axis=0) + 10], self.site_roi_y])

        # Adjust site_roi_x relative to roi_x
        if self.site_roi_x is not None:
            self.site_roi_x = self.site_roi_x - self.roi_x[0]

        # Create visualization rectangles
        self.rect = self._create_site_rectangles()

    def _create_site_rectangles(self):
        """Create rectangle patches for visualization."""
        rect = []
        for i in np.arange(self.site_roi_x.shape[0]):
            rect.append(patches.Rectangle(
                (self.site_roi_x[i,0], self.site_roi_y[i,0]),
                self.site_roi_x[i,1]-self.site_roi_x[i,0],
                self.site_roi_y[i,1]-self.site_roi_y[i,0],
                linewidth=1,
                edgecolor='r',
                facecolor='none',
                alpha=0.3,
                fill=False))
        return rect

    @staticmethod
    def get_h5_path(lyse):
        """Get H5 file path based on lyse session."""
        if lyse.spinning_top:
            return lyse.path
        else:
            df = lyse.data()
            return df.filepath.iloc[-1]

    @staticmethod
    def load_h5_data(h5_path):
        """Load data from H5 file."""
        with h5py.File(h5_path, mode='r+') as f:
            globals_dict = hz.attributesToDictionary(f['globals'])
            info_dict = hz.getAttributeDict(f)
            images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
            kinetix_roi_row = np.array(f['globals'].attrs.get('kinetix_roi_row'))

            # Find looping global variable
            loop_glob = next((glob for group in globals_dict
                            for glob in globals_dict[group]
                            if globals_dict[group][glob][0:2] == "np"), None)

            try:
                loop_var = float(f['globals'].attrs.get(loop_glob))
            except:
                loop_var = info_dict.get('run number')

            return images, kinetix_roi_row, loop_var, info_dict

    def process_images(self, first_image, second_image, first_image_bkg, second_image_bkg):
        """Process image pairs with background subtraction."""
        try:
            sub_image1 = first_image - first_image_bkg
            sub_image2 = second_image - second_image_bkg
        except:
            raise ValueError('Error in background subtraction!')

        # Extract regions of interest for signal and background
        tweezer_roi_1 = sub_image1[:, self.roi_config['x'][0]:self.roi_config['x'][1]]
        bkg_roi_1 = sub_image1[:, self.roi_config['background_x'][0]:self.roi_config['background_x'][1]]

        tweezer_roi_2 = sub_image2[:, self.roi_config['x'][0]:self.roi_config['x'][1]]
        bkg_roi_2 = sub_image2[:, self.roi_config['background_x'][0]:self.roi_config['background_x'][1]]

        return tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2

    def analyze_site_signals(self, tweezer_roi, threshold):
        """Analyze signals at each tweezer site."""
        rect_sig = []
        atom_exist_lst = []
        roi_number_lst = []

        for i in np.arange(self.site_roi_x.shape[0]):
            y_start, y_end = self.site_roi_y[i,0], self.site_roi_y[i,1]
            x_start, x_end = self.site_roi_x[i,0], self.site_roi_x[i,1]

            site_roi_signal = tweezer_roi[y_start:y_end, x_start:x_end]
            signal_sum = np.sum(site_roi_signal)
            roi_number_lst.append(signal_sum)

            if signal_sum > threshold:
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

        return rect_sig, atom_exist_lst, roi_number_lst

    def plot_results(self, tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                    rect_sig_1, rect_sig_2):
        """Plot the analysis results."""
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 20), constrained_layout=True)
        ax_tweezer_1, ax_tweezer_2, ax_bkg_1, ax_bkg_2 = axs

        plt.rcParams.update({'font.size': 14})
        fig.suptitle('Tweezer Array Imaging Analysis', fontsize=16)

        for ax in axs:
            ax.set_xlabel('x [px]', fontsize=14)
            ax.set_ylabel('y [px]', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

        roi_image_scale = 150
        roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

        # Plot first tweezer ROI
        ax_tweezer_1.set_title('1st Tweezer ROI', fontsize=14, pad=10)
        pos = ax_tweezer_1.imshow(tweezer_roi_1, **roi_img_color_kw)
        if self.show_site_roi:
            pc = PatchCollection(self.rect, match_original=True, alpha=0.3)
            ax_tweezer_1.add_collection(pc)
            if rect_sig_1:
                pc_sig = PatchCollection(rect_sig_1, match_original=True)
                ax_tweezer_1.add_collection(pc_sig)
        fig.colorbar(pos, ax=ax_tweezer_1).ax.tick_params(labelsize=12)

        # Plot second tweezer ROI
        ax_tweezer_2.set_title('2nd Tweezer ROI', fontsize=14, pad=10)
        pos = ax_tweezer_2.imshow(tweezer_roi_2, **roi_img_color_kw)
        if self.show_site_roi:
            pc = PatchCollection(self.rect, match_original=True, alpha=0.3)
            ax_tweezer_2.add_collection(pc)
            if rect_sig_2:
                pc_sig = PatchCollection(rect_sig_2, match_original=True)
                ax_tweezer_2.add_collection(pc_sig)
        fig.colorbar(pos, ax=ax_tweezer_2).ax.tick_params(labelsize=12)

        # Plot background ROIs
        ax_bkg_1.set_title('1st Background ROI', fontsize=14, pad=10)
        pos = ax_bkg_1.imshow(bkg_roi_1, **roi_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_1).ax.tick_params(labelsize=12)

        ax_bkg_2.set_title('2nd Background ROI', fontsize=14, pad=10)
        pos = ax_bkg_2.imshow(bkg_roi_2, **roi_img_color_kw)
        fig.colorbar(pos, ax=ax_bkg_2).ax.tick_params(labelsize=12)

    @staticmethod
    def calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2, method='default') -> tuple[float, float]:
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
        n_initial_atoms = np.sum(atom_exist_lst_1)
        survivors = sum(1 for x,y in zip(atom_exist_lst_1, atom_exist_lst_2)
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

    @staticmethod
    def prepare_roi_data(roi_number_lst_1, roi_number_lst_2):
        """Prepare ROI data for saving."""
        roi_number_lst = np.row_stack([np.array(roi_number_lst_1), np.array(roi_number_lst_2)])
        return roi_number_lst.reshape((roi_number_lst.shape[0], roi_number_lst.shape[1], 1))

    @staticmethod
    def save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number):
        """Save analysis results to files."""
        count_file_path = os.path.join(folder_path, 'data.csv')
        roi_number_lst_file_path = os.path.join(folder_path, 'roi_number_lst.npy')

        if run_number == 1:  # Initialize files for first run
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            np.save(roi_number_lst_file_path, roi_number_lst)
        else:  # Append to existing files
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            roi_number_lst_old = np.load(roi_number_lst_file_path)
            roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
            np.save(roi_number_lst_file_path, roi_number_lst_new)

    @staticmethod
    def save_images(folder_path, first_image, second_image, run_number):
        """Save image data to files."""
        np.save(os.path.join(folder_path, 'first'), first_image)
        np.save(os.path.join(folder_path, 'seconds'), second_image)

        count_file_path = os.path.join(folder_path, 'data.csv')
        if run_number == 0:
            with open(count_file_path, 'w') as f_object:
                f_object.write('')
        else:
            with open(count_file_path, 'a') as f_object:
                f_object.write('')


class AverageBackgroundAnalyzer(TweezerImageAnalyzer):
    """Analyzer for average background subtraction method."""

    def __init__(self, roi_config, roi_paths, show_site_roi=True, load_roi=True):
        super().__init__(roi_config, roi_paths, show_site_roi, load_roi)

    def load_threshold(self, load_threshold=True, default_threshold=746.8):
        """Load threshold value from file or use default."""
        if load_threshold:
            threshold_path = os.path.join(self.multishot_path, "th.npy")
            try:
                threshold = np.load(threshold_path)[0]
                # print(f'threshold = {threshold} count')
            except FileNotFoundError:
                print(f'Warning: Threshold file not found at {threshold_path}, using default value')
                threshold = default_threshold
        else:
            threshold = default_threshold
        return threshold

    @staticmethod
    def save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number):
        """Save analysis results to files."""
        count_file_path = os.path.join(folder_path, 'data.csv')
        roi_number_lst_file_path = os.path.join(folder_path, 'roi_number_lst.npy')

        if run_number == 0:  # Initialize files for first run
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            np.save(roi_number_lst_file_path, roi_number_lst)
        else:  # Append to existing files
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            roi_number_lst_old = np.load(roi_number_lst_file_path)
            roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
            np.save(roi_number_lst_file_path, roi_number_lst_new)

    @staticmethod
    def load_average_background(h5_path):
        """Load pre-calculated average background images."""
        avg_shot_bkg_file_path = Path(Path(h5_path).parent.parent, 'avg_shot_bkg.npy')
        try:
            avg_shot_bkg = np.load(avg_shot_bkg_file_path)
            return avg_shot_bkg[0,:,:], avg_shot_bkg[1,:,:]
        except FileNotFoundError:
            raise FileNotFoundError('Make sure you already have the averaged background!')

    def process_run(self, h5_path, load_threshold=True):
        """Process a single run with average background method."""
        folder_path = str(Path(h5_path).parent)

        # Load data and configuration
        images, _, loop_var, info_dict = self.load_h5_data(h5_path)
        threshold = self.load_threshold(load_threshold)
        first_image_bkg, second_image_bkg = self.load_average_background(h5_path)

        # Process images
        image_types = list(images.keys())
        first_image = images[image_types[0]]
        second_image = images[image_types[1]]

        # Process images using loaded ROI values
        tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = self.process_images(
            first_image, second_image, first_image_bkg, second_image_bkg)

        # Analyze signals at each site
        rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = self.analyze_site_signals(
            tweezer_roi_1, threshold)
        rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = self.analyze_site_signals(
            tweezer_roi_2, threshold)

        # Plot results
        self.plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                         rect_sig_1, rect_sig_2)

        # Calculate and save results
        survival_rate, survival_rate_uncert = self.calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)
        roi_number_lst = self.prepare_roi_data(roi_number_lst_1, roi_number_lst_2)
        self.save_data(folder_path, survival_rate, loop_var, roi_number_lst,
                      info_dict.get('run number'))

        # Save values for MLOOP
        # Save sequence analysis result in latest run
        run = lyse.Run(h5_path=h5_path)
        my_condition = True
        # run.save_result(name='survival_rate', value=survival_rate if my_condition else np.nan)

        run.save_results_dict(
            {
                'survival_rate': self.calculate_survival_rate(
                    atom_exist_lst_1, atom_exist_lst_2, method='laplace',
                ) if my_condition else (np.nan, np.nan),
            },
            uncertainties=True,
        )


class AlternatingBackgroundAnalyzer(TweezerImageAnalyzer):
    """Analyzer for alternating background subtraction method."""

    def __init__(self, roi_config, roi_paths, threshold=1185.5, show_site_roi=True, load_roi=True):
        super().__init__(roi_config, roi_paths, show_site_roi, load_roi)
        self.load_roi = load_roi
        self.threshold = threshold

    def process_run(self, h5_path, images):
        """
        Process a single run using the alternating background subtraction method.

        This method implements a two-shot background subtraction where:
        - Even-numbered runs: Save images to be used as background
        - Odd-numbered runs: Use saved images as background and perform analysis

        Processing Steps:
        1. Initial Setup:
           - Determines run number and processing mode (save or analyze)
           - For even runs, saves images and exits
           - For odd runs, continues with analysis

        2. Background Processing (odd runs only):
           - Uses current images as background
           - Loads previous (even) run images as signal images
           - Performs background subtraction

        3. ROI Analysis (odd runs only):
           - Extracts tweezer and background ROIs
           - Analyzes each site using fixed threshold
           - Creates visualization rectangles for detected atoms

        4. Results Processing (odd runs only):
           - Plots processed ROIs with detection overlays
           - Calculates survival rate
           - Saves analysis results to files

        Args:
            h5_path (str): Path to the H5 file containing the run data
            images (dict): Dictionary containing the image data from H5 file

        Note:
            This method requires pairs of runs to work properly:
            - Even run N: Saves images for background
            - Odd run N+1: Uses run N images as background
        """
        folder_path = str(Path(h5_path).parent)
        image_types = list(images.keys())
        first_image = images[image_types[0]]
        second_image = images[image_types[1]]

        # Get run information
        _, _, loop_var, info_dict = self.load_h5_data(h5_path)
        run_number = info_dict.get('run number')

        # Process even-numbered runs (save images for background subtraction)
        if run_number % 2 == 0:
            self.save_images(folder_path, first_image, second_image, run_number)
            return

        # Process odd-numbered runs (perform background subtraction and analysis)
        first_image_bkg = first_image
        second_image_bkg = second_image
        first_image = np.load(os.path.join(folder_path, 'first.npy'))
        second_image = np.load(os.path.join(folder_path, 'seconds.npy'))

        # Process and analyze images
        tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = self.process_images(
            first_image, second_image, first_image_bkg, second_image_bkg)
        if self.load_roi is True:
            rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = self.analyze_site_signals(
                tweezer_roi_1, self.threshold)
            rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = self.analyze_site_signals(
                tweezer_roi_2, self.threshold)
        else:
            rect_sig_1 = None
            rect_sig_2 = None

        # Plot results
        self.plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                         rect_sig_1, rect_sig_2)

        # Calculate and save results
        if self.load_roi is True:
            survival_rate, _ = self.calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)
            roi_number_lst = self.prepare_roi_data(roi_number_lst_1, roi_number_lst_2)
            self.save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number)

class AverageBackground2DScanAnalyzer(AverageBackgroundAnalyzer):
    """Extended analyzer for 2D parameter scans."""

    @staticmethod

    def load_h5_data(h5_path):
        """Load data from H5 file."""
        with h5py.File(h5_path, mode='r+') as f:
            globals_dict = hz.attributesToDictionary(f['globals'])
            images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)

            # Extract looping globals
            loop_globals = []
            for group in globals_dict:
                for glob in globals_dict[group]:
                    if globals_dict[group][glob][0:2] == "np":
                        loop_globals.append(float(hz.attributesToDictionary(f).get('globals').get(glob)))

            if len(loop_globals) != 2:
                raise ValueError("Expected exactly 2 looping globals")

            loop_var_1 = float(loop_globals[0])
            loop_var_2 = float(loop_globals[1])
            info_dict = hz.getAttributeDict(f)
            kinetix_roi_row= np.array(f['globals'].attrs.get('kinetix_roi_row'))

            return images, kinetix_roi_row, loop_var_1, loop_var_2, info_dict

    def process_run(self, h5_path, load_threshold=True):
        """Process a single run with average background method."""
        folder_path = str(Path(h5_path).parent)

        # Load data and configuration
        images, _, loop_var_1, loop_var_2, info_dict = self.load_h5_data(h5_path)
        threshold = self.load_threshold(load_threshold)
        first_image_bkg, second_image_bkg = self.load_average_background(h5_path)

        # Process images
        image_types = list(images.keys())
        first_image = images[image_types[0]]
        second_image = images[image_types[1]]

        # Process images using loaded ROI values
        tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = self.process_images(
            first_image, second_image, first_image_bkg, second_image_bkg)

        # Analyze signals at each site
        rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = self.analyze_site_signals(
            tweezer_roi_1, threshold)
        rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = self.analyze_site_signals(
            tweezer_roi_2, threshold)

        # Plot results
        self.plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                         rect_sig_1, rect_sig_2)

        # Calculate and save results
        survival_rate, survival_rate_uncert = self.calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)
        roi_number_lst = self.prepare_roi_data(roi_number_lst_1, roi_number_lst_2)
        self.save_data(folder_path, survival_rate, roi_number_lst,
                      info_dict.get('run number'),  loop_var_1, loop_var_2)

        # Save values for MLOOP
        # Save sequence analysis result in latest run
        run = lyse.Run(h5_path=h5_path)
        my_condition = True
        # run.save_result(name='survival_rate', value=survival_rate if my_condition else np.nan)
        # run.save_result(name='u_survival_rate', value=survival_rate_uncert if my_condition else np.nan)
        run.save_results_dict(
            {
                'survival_rate': self.calculate_survival_rate(
                    atom_exist_lst_1, atom_exist_lst_2, method='laplace',
                ) if my_condition else (np.nan, np.nan),
            },
            uncertainties=True,
        )

    def save_data(self, folder_path,survival_rate, roi_number_lst, run_number, param1, param2):
        """Save analysis results to files."""
        count_file_path = os.path.join(folder_path, 'data.csv')
        roi_number_lst_file_path = os.path.join(folder_path, 'roi_number_lst.npy')
        if run_number == 0: # or run_number == 1:  # Initialize files for first run
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{survival_rate},{param1},{param2}\n')
            np.save(roi_number_lst_file_path, roi_number_lst)
        else:  # Append to existing files
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{survival_rate},{param1},{param2}\n')
            roi_number_lst_old = np.load(roi_number_lst_file_path)
            roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
            print(roi_number_lst_new.shape)
            np.save(roi_number_lst_file_path, roi_number_lst_new)
