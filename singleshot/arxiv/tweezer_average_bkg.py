"""
Tweezer imaging analysis script with average background subtraction for Kinetix camera.
This script processes tweezer imaging data using a pre-calculated average background.

Created on: 2024-12-26
"""

import os
import sys

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
    'y': [960, 960+110],   # Vertical region of interest
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

    # Get H5 file path and process the run
    h5_path = analyzer.get_h5_path(lyse)
    print(h5_path)
    analyzer.process_run(h5_path, LOAD_THRESHOLD)

if __name__ == "__main__":
    main()
