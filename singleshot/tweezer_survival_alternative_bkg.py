from analysis_lib import TweezerAnalysis, TweezerPlotter, AnalysisConfig
import matplotlib.pyplot as plt

# ROI config path: use when load_roi = False
roi_config_path = 'X:\\userlib\\analysislib\\scripts\\multishot\\tweezer_roi.yaml'

analysis_config = AnalysisConfig.from_yaml(roi_config_path)

# Initialize analysis with background ROI and standard ROI loading
tweezer_analyzer = TweezerAnalysis(
    config=analysis_config,
    load_type='lyse',
    h5_path=None
)

tweezer_plotter = TweezerPlotter(
    tweezer_analyzer
)

# plot results
tweezer_plotter.plot_images(roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True)
