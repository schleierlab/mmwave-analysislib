import sys
import os

# Add analysis library root to Python path
root_path = r"X:\userlib\analysislib"
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    import lyse
except ImportError:
    from analysis.data import h5lyze as lyse

from singleshot.common.tweezer_analysis import TweezerAnalysis
from singleshot.common.tweezer_plot import TweezerPlotter
from singleshot.common.analysis_config import TweezerAnalysisConfig, kinetix_system
from singleshot.common.plot_config import PlotConfig

analysis_config = TweezerAnalysisConfig()
analysis_config.method = 'average'

# Initialize analysis with background ROI and standard ROI loading
tweezer_analyzer = TweezerAnalysis(
    config=analysis_config,
    load_type='lyse',
    h5_path=None
)

# Initialize plotter with consistent styling
tweezer_plotter = TweezerPlotter(
    tweezer_analyzer,
    plot_config=PlotConfig()
)

# plot results
tweezer_plotter.plot_images(show_site_roi=True, plot_bkg_roi=True)
