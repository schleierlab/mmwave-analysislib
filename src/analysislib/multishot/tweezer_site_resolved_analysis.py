from pathlib import Path

from analysislib.common.plot_config import PlotConfig
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.multishot.util import select_data_directory


folder = select_data_directory()
processed_results_fname = Path(folder) / TweezerPreprocessor.PROCESSED_RESULTS_FNAME
tweezer_statistician = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
    plot_config=PlotConfig(),
)
tweezer_statistician.plot_survival_rate_by_site_2d()
