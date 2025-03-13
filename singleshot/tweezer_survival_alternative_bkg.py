from .tweezer_analysis import TweezerAnalysis, TweezerPlotter
import matplotlib.pyplot as plt
from .analysis_config import kinetix_system, TweezerAnalysisConfig
from .plot_config import PlotConfig

analysis_config = TweezerAnalysisConfig()
analysis_config.method = 'alternative'

# Initialize analysis with background ROI and standard ROI loading
tweezer_analyzer = TweezerAnalysis(
    config=analysis_config,
    load_type='lyse',
    h5_path=None
)

tweezer_plotter = TweezerPlotter(
    tweezer_analyzer,
    plot_config=PlotConfig()
)

# plot results
tweezer_plotter.plot_images(roi_image_scale=150, show_site_roi=True, plot_bkg_roi=True)
