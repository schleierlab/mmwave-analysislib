"""
Tweezer imaging analysis script with alternating background for Kinetix camera.
This script processes tweezer imaging data with background subtraction.
This is a refactored version that uses the tweezer_imaging_util module.

Created on: 2024-12-26
"""

import sys
import os
import numpy as np
import matplotlib.patches as patches
import csv

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

try:
    lyse
except NameError:
    import lyse

from scripts.singleshot.tweezer_imaging_util import (
    load_h5_data, load_roi_data, analyze_site_signals,
    plot_results, process_images
)

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

def main():
    # Is this script being run from within an interactive lyse session?
    if lyse.spinning_top:
        h5_path = lyse.path
    else:
        df = lyse.data()
        h5_path = df.filepath.iloc[-1]

    images, kinetix_roi_row, loop_var, info_dict = load_h5_data(h5_path)
    site_roi_x, site_roi_y, roi_x = load_roi_data(ROI_PATHS, LOAD_ROI)

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
            first_image, second_image, first_image_bkg, second_image_bkg, ROI_CONFIG)

        rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = analyze_site_signals(
            tweezer_roi_1, site_roi_x, site_roi_y, THRESHOLD)
        rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = analyze_site_signals(
            tweezer_roi_2, site_roi_x, site_roi_y, THRESHOLD)

        plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2, 
                    rect, rect_sig_1, rect_sig_2, SHOW_SITE_ROI)

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
