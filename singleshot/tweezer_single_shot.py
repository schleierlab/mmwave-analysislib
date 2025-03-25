import matplotlib.pyplot as plt

from analysislib.scripts.common.tweezer_analysis import TweezerPreprocessor
from analysislib.scripts.common.tweezer_statistics import TweezerStatistician
from analysislib.scripts.common.plot_config import PlotConfig


# Initialize analysis with background ROI and standard ROI loading
tweezer_analyzer = TweezerPreprocessor(
    load_type='lyse',
    h5_path=None,
)

fig = plt.figure(layout='constrained', figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

processed_results_fname = tweezer_analyzer.process_shot()
tweezer_analyzer.show_image(roi_patches=True, fig=subfigs[0], vmax=70)

# Initialize statistician with consistent styling
tweezer_statistician = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=tweezer_analyzer.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)
#tweezer_statistician.plot_survival_rate(fig=subfigs[1])
tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
