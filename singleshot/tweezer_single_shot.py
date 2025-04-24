import matplotlib.pyplot as plt

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig

SHOW_ROIS = True
FIT_LORENTZ = False

# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse',
    h5_path=None,
    use_averaged_background=True
)

fig = plt.figure(layout='constrained', figsize=(10, 4))
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=True)
tweezer_preproc.show_image(roi_patches=SHOW_ROIS, fig=subfigs[0], vmax=100)

# Initialize statistician with consistent styling
tweezer_statistician = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)
tweezer_statistician.plot_survival_rate(fig=subfigs[1], plot_lorentz = FIT_LORENTZ)

#tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
