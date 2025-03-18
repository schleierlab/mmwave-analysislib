from .common.tweezer_analysis import TweezerPreprocessor
from .common.analysis_config import kinetix_system


# ROI config path: use when load_roi = False
roi_config_path = R'X:\userlib\analysislib\scripts\multishot\tweezer_roi.yaml'

# use when load_threshold = False
threshold = 1000

# Initialize analysis with background ROI and standard ROI loading
tweezer_analysis_obj = TweezerPreprocessor(
    imaging_setup=kinetix_system,
    method='alternating',  # Use alternating background subtraction
    bkg_roi_x=[1900, 2400],  # Background ROI x-coordinates
    load_roi=True,  # If True, load roi_x and site ROIs from standard .npy files
    roi_config_path= None,
    load_threshold=True,
    threshold=None, # If load_threshold is False, provide threshold
)

# plot results
tweezer_analysis_obj.plot_images(roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True)
