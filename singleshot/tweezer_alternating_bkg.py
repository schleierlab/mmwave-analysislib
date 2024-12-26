"""
Tweezer imaging analysis script with alternating background for Kinetix camera.
This script processes tweezer imaging data with alternating background subtraction.

Created on Thu Feb 2 15:11:12 2023
Updated on Dec 26 2024
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

try:
    lyse
except NameError:
    import lyse

from scripts.singleshot.tweezer_imaging_util import (
    get_h5_path, load_h5_data, load_roi_data, analyze_site_signals,
    plot_results, process_images, create_site_rectangles, calculate_survival_rate,
    prepare_roi_data, save_data, save_images
)

# Configuration Constants
SHOW_SITE_ROI = True
LOAD_ROI = True
THRESHOLD = 1185.5  # Fixed threshold for alternating background method

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
    # Get H5 file path
    h5_path = get_h5_path(lyse)
    folder_path = str(Path(h5_path).parent)

    # Load data and configuration
    images, kinetix_roi_row, loop_var, info_dict = load_h5_data(h5_path)
    site_roi_x, site_roi_y, roi_x = load_roi_data(ROI_PATHS, LOAD_ROI)
    run_number = info_dict.get('run number')

    if site_roi_x is not None:
        site_roi_x = site_roi_x - roi_x[0]

    # Create visualization rectangles
    rect = create_site_rectangles(site_roi_x, site_roi_y)

    image_types = list(images.keys())
    first_image = images[image_types[0]]
    second_image = images[image_types[1]]

    # Process even-numbered runs (save images for background subtraction)
    if run_number % 2 == 0:
        save_images(folder_path, first_image, second_image, run_number)
        return

    # Process odd-numbered runs (perform background subtraction and analysis)
    first_image_bkg = first_image
    second_image_bkg = second_image
    first_image = np.load(os.path.join(folder_path, 'first.npy'))
    second_image = np.load(os.path.join(folder_path, 'seconds.npy'))

    # Process images using loaded ROI values
    tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = process_images(
        first_image, second_image, first_image_bkg, second_image_bkg, ROI_CONFIG)

    # Analyze signals at each site
    rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = analyze_site_signals(
        tweezer_roi_1, site_roi_x, site_roi_y, THRESHOLD)
    rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = analyze_site_signals(
        tweezer_roi_2, site_roi_x, site_roi_y, THRESHOLD)

    # Plot results
    plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2, 
                rect, rect_sig_1, rect_sig_2, SHOW_SITE_ROI)

    # Calculate survival rate
    survival_rate = calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)

    # Prepare and save ROI data
    roi_number_lst = prepare_roi_data(roi_number_lst_1, roi_number_lst_2)

    # Save results
    save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number)

if __name__ == "__main__":
    main()
