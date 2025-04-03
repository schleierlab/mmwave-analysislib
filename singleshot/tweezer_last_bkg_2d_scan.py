"""
Tweezer image analysis script for 2D parameter scan with average background subtraction.
This script processes tweezer imaging data with two looping global parameters.

Created on: 2024-12-26
"""

import sys


# Add analysis lib to path
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from analysis.data import h5lyze as hz
import os

try:
    lyse
except NameError:
    import lyse

from analysislib.scripts.singleshot.arxiv.tweezer_image_analysis import TweezerImageAnalyzer, AverageBackgroundAnalyzer, LastBackground2DScanAnalyzer

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
    analyzer = LastBackground2DScanAnalyzer(
        roi_config=ROI_CONFIG,
        roi_paths=ROI_PATHS,
        show_site_roi=True,
        load_roi=True
    )


    # Get H5 file path and process the run
    h5_path = analyzer.get_h5_path(lyse)
    analyzer.process_run(h5_path, LOAD_THRESHOLD)


if __name__ == '__main__':
    main()
