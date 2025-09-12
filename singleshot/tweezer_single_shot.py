import matplotlib.pyplot as plt
import os

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig
import numpy as np

SHOW_ROIS = True
SHOW_INDEX = True # site index will not show up if show_rois is set to false
USE_AVERAGED_BACKGROUND = True
FIT_TYPE_1D = None # do a curve fit at the final shot, set to None when don't to curve fit
# options: 'lorentzian',

EXACT_REARRANGEMENT = False
PLOT_PAIR_STATES = False
SHOW_IMG_ONLY = False
SAVE_DATA_CSV_FILE = False # need to be False for 2d scans!


# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse',
    h5_path=None,
    use_averaged_background = USE_AVERAGED_BACKGROUND
)
doing_rearrangement = bool(tweezer_preproc.parameters['do_rearrangement'])
print('do rearrangement: ', doing_rearrangement)

fig = plt.figure(layout='constrained')
processed_results_fname = tweezer_preproc.process_shot(use_global_threshold = True)
if SHOW_IMG_ONLY:
    tweezer_preproc.show_image(roi_patches=SHOW_ROIS, site_index = SHOW_INDEX, fig=fig, vmax=80)
else:
    subfigs = fig.subfigures(nrows=1, ncols=3, wspace=0.07)
    tweezer_preproc.show_image(roi_patches=SHOW_ROIS, site_index = SHOW_INDEX, fig=subfigs[0], vmax=80)

# Initialize statistician with consistent styling
tweezer_statistician = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
    rearrangement=doing_rearrangement,
    target_sites=tweezer_preproc.target_array,
)

folder_path = os.path.dirname(tweezer_preproc.h5_path)
if not SHOW_IMG_ONLY:
    if SAVE_DATA_CSV_FILE:
        indep_var, survival_rates, survival_rate_errs = tweezer_statistician.plot_survival_rate(fig=subfigs[1], fit_type_1d = FIT_TYPE_1D, require_exact_rearrangement=EXACT_REARRANGEMENT, plot_pair_states = PLOT_PAIR_STATES)
        np.savetxt(folder_path + "/data.csv", [indep_var, survival_rates, survival_rate_errs], delimiter=",")
    else:
        tweezer_statistician.plot_survival_rate(fig=subfigs[1], fit_type_1d = FIT_TYPE_1D, require_exact_rearrangement=EXACT_REARRANGEMENT)

    tweezer_statistician.plot_tweezing_statistics(fig=subfigs[2])

    # TODO: this function right now doesn't work with 2d parameter scan

if tweezer_statistician.is_final_shot:
    figname = folder_path + '/tweezer_single_shot.pdf'
    fig.savefig(figname)

#tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
