import matplotlib.pyplot as plt
import os

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig

SHOW_ROIS = True
SHOW_INDEX = True # site index will not show up if show_rois is set to false
FIT_LORENTZ = False
USE_AVERAGED_BACKGROUND = True
SHOW_IMG_ONLY = False

# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse',
    h5_path=None,
    use_averaged_background = USE_AVERAGED_BACKGROUND
)
doing_rearrangement = bool(tweezer_preproc.parameters['do_rearrangement'])

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
    tweezer_statistician.plot_survival_rate(fig=subfigs[1], plot_lorentz = FIT_LORENTZ)
    tweezer_statistician.plot_tweezing_statistics(fig=subfigs[2])

        # print(unique_params)
        # print(survival_rates)
        # print(sigma_beta)
        # np.savetxt(folder_path + "/data.csv", [unique_params, survival_rates, sigma_beta], delimiter=",")
        # TODO: this function right now doesn't work with 2d parameter scan

figname = folder_path + '/tweezer_single_shot.pdf'
fig.savefig(figname)
#tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
