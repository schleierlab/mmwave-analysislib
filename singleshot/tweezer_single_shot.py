import matplotlib.pyplot as plt

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig

SHOW_ROIS = True
SHOW_INDEX = True # site index will not show up if show_rois is set to false
FIT_LORENTZ = False
USE_AVERAGED_BACKGROUND = False
SHOW_IMG_ONLY = False

# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse',
    h5_path=None,
    use_averaged_background = USE_AVERAGED_BACKGROUND
)

fig = plt.figure(layout='constrained')
processed_results_fname = tweezer_preproc.process_shot(use_global_threshold = True)
if SHOW_IMG_ONLY:
    tweezer_preproc.show_image(roi_patches=SHOW_ROIS, site_index = SHOW_INDEX, fig=fig, vmax=100)
else:
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)
    tweezer_preproc.show_image(roi_patches=SHOW_ROIS, site_index = SHOW_INDEX, fig=subfigs[0], vmax=100)

target_array = tweezer_preproc.target_array

# Initialize statistician with consistent styling
tweezer_statistician = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)

if not SHOW_IMG_ONLY:
    if bool(tweezer_preproc.globals['do_rearrangement']):
        tweezer_statistician.plot_target_sites_success_rate(target_array, fig = subfigs[1])
    else:
        tweezer_statistician.plot_survival_rate(fig=subfigs[1], plot_lorentz = FIT_LORENTZ)

#tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
