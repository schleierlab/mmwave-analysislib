"""
Tweezer imaging analysis script with alternating background for Kinetix camera.
This script processes tweezer imaging data with background subtraction.

Created on Thu Feb 2 15:11:12 2023
Updated on Dec 19 2024
"""

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import csv

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

try:
    lyse
except NameError:
    import lyse

from analysis.data import h5lyze as hz

# Configuration Constants
SHOW_SITE_ROI = True
LOAD_ROI = True
THRESHOLD = 1185.5

# Imaging Parameters
PIXEL_SIZE = 1
MAGNIFICATION = 1
COUNTS_PER_ATOM = 1

# ROI Configuration
ROI_CONFIG = {
    'x': [1173, 1523],  # Region of interest for tweezer array
    'y': [960, 1070],   # Vertical region of interest
    'background_x': [1900, 2400],
    'background_y': [1900, 2400]
}

# File paths
FOLDER_PATH = 'X:\\userlib\\analysislib\\scripts\\multishot\\'
ROI_PATHS = {
    'site_roi_x': os.path.join(FOLDER_PATH, "site_roi_x.npy"),
    'site_roi_y': os.path.join(FOLDER_PATH, "site_roi_y.npy"),
    'roi_x': os.path.join(FOLDER_PATH, "roi_x.npy")
}

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

def load_roi_data():
    """Load ROI data from files."""
    if not LOAD_ROI:
        return None, None, None
        
    site_roi_x = np.load(ROI_PATHS['site_roi_x'])
    site_roi_y = np.load(ROI_PATHS['site_roi_y'])
    roi_x = np.load(ROI_PATHS['roi_x'])
    
    # Add minimum value as first entry and adjust y-offset
    site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], site_roi_x])
    site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) + 10], site_roi_y])
    
    return site_roi_x, site_roi_y, roi_x

def analyze_site_signals(tweezer_roi, site_roi_x, site_roi_y):
    """Analyze signals at each tweezer site."""
    rect_sig = []
    atom_exist_lst = []
    roi_number_lst = []
    
    for i in np.arange(site_roi_x.shape[0]):
        y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
        x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]
        
        site_roi_signal = tweezer_roi[y_start:y_end, x_start:x_end]
        signal_sum = np.sum(site_roi_signal)
        roi_number_lst.append(signal_sum)
        
        if signal_sum > THRESHOLD:
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

def plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2, 
                rect, rect_sig_1, rect_sig_2):
    """Plot the analysis results."""
    # Create figure with larger size
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 20), constrained_layout=True)
    ax_tweezer_1, ax_tweezer_2, ax_bkg_1, ax_bkg_2 = axs

    # Set larger font sizes
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
    if SHOW_SITE_ROI:
        pc = PatchCollection(rect, match_original=True, alpha=0.3)
        ax_tweezer_1.add_collection(pc)
        if rect_sig_1:
            pc_sig = PatchCollection(rect_sig_1, match_original=True)
            ax_tweezer_1.add_collection(pc_sig)
    cbar = fig.colorbar(pos, ax=ax_tweezer_1)
    cbar.ax.tick_params(labelsize=12)

    # Plot second tweezer ROI
    ax_tweezer_2.set_title('2nd Tweezer ROI', fontsize=14, pad=10)
    pos = ax_tweezer_2.imshow(tweezer_roi_2, **roi_img_color_kw)
    if SHOW_SITE_ROI:
        pc = PatchCollection(rect, match_original=True, alpha=0.3)
        ax_tweezer_2.add_collection(pc)
        if rect_sig_2:
            pc_sig = PatchCollection(rect_sig_2, match_original=True)
            ax_tweezer_2.add_collection(pc_sig)
    cbar = fig.colorbar(pos, ax=ax_tweezer_2)
    cbar.ax.tick_params(labelsize=12)

    # Plot first background ROI
    ax_bkg_1.set_title('1st Background ROI', fontsize=14, pad=10)
    pos = ax_bkg_1.imshow(bkg_roi_1, **roi_img_color_kw)
    cbar = fig.colorbar(pos, ax=ax_bkg_1)
    cbar.ax.tick_params(labelsize=12)

    # Plot second background ROI
    ax_bkg_2.set_title('2nd Background ROI', fontsize=14, pad=10)
    pos = ax_bkg_2.imshow(bkg_roi_2, **roi_img_color_kw)
    cbar = fig.colorbar(pos, ax=ax_bkg_2)
    cbar.ax.tick_params(labelsize=12)

def process_images(first_image, second_image, first_image_bkg, second_image_bkg, roi_x):
    """Process image pairs with background subtraction."""
    try:
        sub_image1 = first_image - first_image_bkg
        sub_image2 = second_image - second_image_bkg
    except:
        print('Start from even run number shots!')
        quit()

    # Extract regions of interest for signal and background
    tweezer_roi_1 = sub_image1[:, roi_x[0]:roi_x[1]]
    bkg_roi_1 = sub_image1[:, ROI_CONFIG['background_x'][0]:ROI_CONFIG['background_x'][1]]

    tweezer_roi_2 = sub_image2[:, roi_x[0]:roi_x[1]]
    bkg_roi_2 = sub_image2[:, ROI_CONFIG['background_x'][0]:ROI_CONFIG['background_x'][1]]

    return tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2

def main():
    # Is this script being run from within an interactive lyse session?
    if lyse.spinning_top:
        h5_path = lyse.path
    else:
        df = lyse.data()
        h5_path = df.filepath.iloc[-1]

    images, kinetix_roi_row, loop_var, info_dict = load_h5_data(h5_path)
    site_roi_x, site_roi_y, roi_x = load_roi_data()

    if site_roi_x is not None:
        site_roi_x = site_roi_x - roi_x[0]

    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]
        y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
        rect.append(patches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=1,
            edgecolor='r',
            facecolor='none',
            alpha=0.3,
            fill=False))

    image_types = list(images.keys())
    pixels = images[image_types[0]].shape[0]
    fov = pixels*PIXEL_SIZE*MAGNIFICATION

    folder_path = '\\'.join(h5_path.split('\\')[0:-1])
    count_file_path = folder_path+'\\data.csv'

    if info_dict.get('run number') % 2 == 0:
        first_image = images[image_types[0]]
        second_image = images[image_types[1]]

        np.save(folder_path+'\\first', first_image)
        np.save(folder_path+'\\seconds', second_image)

        if info_dict.get('run number') == 0:
            with open(count_file_path, 'w') as f_object:
                f_object.write('')
        else:
            with open(count_file_path, 'a') as f_object:
                f_object.write('')
    else:
        first_image_bkg = images[image_types[0]]
        second_image_bkg = images[image_types[1]]
        first_image = np.load(folder_path+'\\first.npy')
        second_image = np.load(folder_path+'\\seconds.npy')

        # Process images using loaded ROI values
        tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = process_images(
            first_image, second_image, first_image_bkg, second_image_bkg, roi_x)

        rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = analyze_site_signals(
            tweezer_roi_1, site_roi_x, site_roi_y)
        rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = analyze_site_signals(
            tweezer_roi_2, site_roi_x, site_roi_y)

        plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2, 
                    rect, rect_sig_1, rect_sig_2)

        atom_exist_lst_1 = np.array(atom_exist_lst_1)
        atom_exist_lst_2 = np.array(atom_exist_lst_2)

        survival_rate = sum(1 for x,y in zip(atom_exist_lst_1,atom_exist_lst_2) if x == 1 and y == 1) / np.sum(atom_exist_lst_1)

        roi_number_lst_file_path = folder_path+'\\roi_number_lst.npy'

        roi_number_lst = np.row_stack([np.array(roi_number_lst_1), np.array(roi_number_lst_2)])

        roi_number_lst = roi_number_lst.reshape((roi_number_lst.shape[0], roi_number_lst.shape[1],1))

        if info_dict.get('run number') == 1:
            with open(count_file_path, 'w') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            np.save(roi_number_lst_file_path, roi_number_lst)

        else:
            with open(count_file_path, 'a') as f_object:
                f_object.write(f'{survival_rate},{loop_var}\n')
            roi_number_lst_old = np.load(roi_number_lst_file_path)
            roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
            np.save(roi_number_lst_file_path, roi_number_lst_new)

if __name__ == "__main__":
    main()
