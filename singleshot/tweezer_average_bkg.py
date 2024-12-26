"""
Tweezer imaging analysis script with average background subtraction for Kinetix camera.
This script processes tweezer imaging data using a pre-calculated average background.

Created on: 2024-12-26
"""

import os
import sys
from pathlib import Path

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

try:
    lyse
except NameError:
    import lyse

from scripts.singleshot.tweezer_image_analysis import AverageBackgroundAnalyzer

# Configuration Constants
SHOW_SITE_ROI = True
LOAD_ROI = True
LOAD_THRESHOLD = True

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
    # Initialize analyzer
    analyzer = AverageBackgroundAnalyzer(ROI_CONFIG, ROI_PATHS, SHOW_SITE_ROI, LOAD_ROI)

    # Get H5 file path
    h5_path = analyzer.get_h5_path(lyse)

    # Load data and configuration
    images, _, loop_var, info_dict = analyzer.load_h5_data(h5_path)
    threshold = analyzer.load_threshold(FOLDER_PATH, LOAD_THRESHOLD)
    first_image_bkg, second_image_bkg = analyzer.load_average_background(h5_path)

    # Process images
    image_types = list(images.keys())
    first_image = images[image_types[0]]
    second_image = images[image_types[1]]

    # Process images using loaded ROI values
    tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2 = analyzer.process_images(
        first_image, second_image, first_image_bkg, second_image_bkg)

    # Analyze signals at each site
    rect_sig_1, atom_exist_lst_1, roi_number_lst_1 = analyzer.analyze_site_signals(
        tweezer_roi_1, threshold)

    print("new code", analyzer.rect[0])
    print(analyzer.site_roi_x[0,0:2])

    rect_sig_2, atom_exist_lst_2, roi_number_lst_2 = analyzer.analyze_site_signals(
        tweezer_roi_2, threshold)

    # Plot results
    analyzer.plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2,
                         rect_sig_1, rect_sig_2)

    # Calculate survival rate
    survival_rate = analyzer.calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2)

    # Prepare and save ROI data
    roi_number_lst = analyzer.prepare_roi_data(roi_number_lst_1, roi_number_lst_2)
    analyzer.save_data(str(Path(h5_path).parent), survival_rate, loop_var,
                      roi_number_lst, info_dict.get('run number'))

if __name__ == "__main__":
    main()
