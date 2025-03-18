from common.tweezer_analysis import TweezerPreprocessor
from common.tweezer_plot import TweezerPlotter
from common.analysis_config import TweezerAnalysisConfig, kinetix_system
from common.plot_config import PlotConfig


analysis_config = TweezerAnalysisConfig(
    imaging_system=kinetix_system,
)

# Initialize analysis with background ROI and standard ROI loading
tweezer_analyzer = TweezerPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
)

processed_results_fname = tweezer_analyzer.process_shot()

# Initialize plotter with consistent styling
tweezer_plotter = TweezerPlotter(
    processed_results_fname,
    plot_config=PlotConfig(),
)
tweezer_plotter.plot_survival_rate()
