"""
Tweezer imaging analysis script with alternating background for Kinetix camera.
This script processes tweezer imaging data with alternating background subtraction.

Created on Thu Feb 2 15:11:12 2023
Updated on Dec 26 2024
"""

import sys
import os

# Add analysis lib to path
ROOT_PATH = r"X:\userlib\analysislib"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

try:
    lyse
except NameError:
    import lyse

from scripts.singleshot.tweezer_image_analysis import AlternatingBackgroundAnalyzer

# Configuration Constants
SHOW_SITE_ROI = False
LOAD_ROI = True
THRESHOLD = 744 #1185.5  # Fixed threshold for alternating background method

# ROI Configuration
ROI_CONFIG = {
    'x': [1050,1500],#[900,1300],  # Region of interest for tweezer array
    'y': [1200,1310],#[1100,1200],   # Vertical region of interest
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
    analyzer = AlternatingBackgroundAnalyzer(
        ROI_CONFIG, ROI_PATHS, THRESHOLD, SHOW_SITE_ROI, LOAD_ROI)

    # Get H5 file path and process the run
    h5_path = analyzer.get_h5_path(lyse)
    images, _, _, _ = analyzer.load_h5_data(h5_path)
    analyzer.process_run(h5_path, images)

if __name__ == "__main__":
    main()
