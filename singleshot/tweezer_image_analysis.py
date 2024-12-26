"""
Class-based implementation of tweezer image analysis functionality.
This module provides a structured approach to analyzing tweezer imaging data
with both average and alternating background subtraction methods.

Created on: 2024-12-26
"""

import os
from pathlib import Path

import h5py
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
        self.roi_config["x"][:]= self.roi_x

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
        print("new code", [self.roi_config['x'][0],self.roi_config['x'][1]])
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
    def calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2):
        """Calculate survival rate from atom existence lists."""
        return sum(1 for x,y in zip(atom_exist_lst_1, atom_exist_lst_2)
                  if x == 1 and y == 1) / np.sum(atom_exist_lst_1)

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

        if run_number == 0 or run_number == 1:  # Initialize files for first run
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

    @staticmethod
    def load_threshold(folder_path, load_threshold=True, default_threshold=746.8):
        """Load threshold value from file or use default."""
        if load_threshold:
            threshold = np.load(os.path.join(folder_path, "th.npy"))[0]
            print(f'threshold = {threshold} count')
        else:
            threshold = default_threshold
        return threshold

    @staticmethod
    def load_average_background(h5_path):
        """Load pre-calculated average background images."""
        avg_shot_bkg_file_path = Path(Path(h5_path).parent.parent, 'avg_shot_bkg.npy')
        try:
            avg_shot_bkg = np.load(avg_shot_bkg_file_path)
            return avg_shot_bkg[0,:,:], avg_shot_bkg[1,:,:]
        except FileNotFoundError:
            raise FileNotFoundError('Make sure you already have the averaged background!')

class AlternatingBackgroundAnalyzer(TweezerImageAnalyzer):
    """Analyzer for alternating background subtraction method."""

    def __init__(self, roi_config, roi_paths, threshold=1185.5, show_site_roi=True, load_roi=True):
        super().__init__(roi_config, roi_paths, show_site_roi, load_roi)
        self.threshold = threshold

    def process_run(self, h5_path, images):
        """Process a single run with alternating background method."""
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

        rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = self.analyze_site_signals(
            tweezer_roi_1, self.threshold)
        rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = self.analyze_site_signals(
            tweezer_roi_2, self.threshold)

        # Plot results
        self.plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                         rect_sig_1, rect_sig_2)

        # Calculate and save results
        survival_rate = self.calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)
        roi_number_lst = self.prepare_roi_data(roi_number_lst_1, roi_number_lst_2)
        self.save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number)
