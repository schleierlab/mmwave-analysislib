
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path
'''
Do tweezer multishot analysis without doing auto roi detection. Plot loading rate, histagrams, imaging fidelity...
Load data directly from 'tweezer_preprocess.h5' (this is the file generated after running tweezer_single_shot) to do multishot analysis.
'''
background_subtract = True
USE_AVERAGED_BACKGROUND = True
PLOT_AVERAGED_IMAGES = False # Show averaged image(s) but takes longer time because need to process through all h5 files
PLOT_TWO_IMAGES = True # only matters if PLOT_AVERAGED_IMAGES is set to True. If set to False, only plot the first image.
folder = select_data_directory()

tweezer_preproc = TweezerPreprocessor(
        load_type='h5',
        h5_path = next(folder.glob('20*0.h5')), # needed h5 file only to look at kinetix_roi_raw -- is there a better way to do this?
    )
new_site_rois = tweezer_preproc.site_rois
preproc_h5_path = Path(folder) / TweezerPreprocessor.PROCESSED_RESULTS_FNAME

if PLOT_AVERAGED_IMAGES:
    finder = TweezerFinder.load_from_h5(folder, use_averaged_background = USE_AVERAGED_BACKGROUND, include_2_images = PLOT_TWO_IMAGES)
    if PLOT_TWO_IMAGES:
        finder.plot_averaged_images(new_site_rois)
    else:
        finder.plot_sites(new_site_rois)

tweezer_statistician = TweezerStatistician(
    preproc_h5_path=preproc_h5_path,
)

thresholder = TweezerThresholder(
    None,
    new_site_rois,
    background_subtract=background_subtract,
    weights=None,
    processed_results_fname=preproc_h5_path
)

thresholder.fit_gmms() # gmm stands for Gaussian mixture model
thresholder.overwrite_thresholds_to_yaml(folder)

# TODO: add function to plot averaged tweezer image
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(f'{folder}')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])
tweezer_statistician.plot_survival_rate_by_site_2d()
tweezer_statistician.plot_avg_survival_rate_by_grouped_sites_1d_old(group_size = 1, fit_type = 'lorentzian')

axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')

fig.savefig(f'{folder}/tweezers_roi_detection.pdf')
