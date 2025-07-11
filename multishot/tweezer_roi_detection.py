
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path
import numpy as np

background_subtract = True
USE_AVERAGED_BACKGROUND = True
folder = select_data_directory()

finder = TweezerFinder.load_from_h5(folder, use_averaged_background = USE_AVERAGED_BACKGROUND)
new_site_rois = finder.detect_rois_by_roi_number(roi_number= 50, neighborhood_size=5, detection_threshold = 35)
finder.overwrite_site_rois_to_yaml(new_site_rois, folder)
# TODO: evaluate whether or not we actually should be subtracting the background for tweezers
# TODO: Include survival rate if taking two shots

multishot_analysis = TweezerMultishotAnalysis(folder, use_averaged_background = USE_AVERAGED_BACKGROUND)

finder.plot_sites(new_site_rois)

thresholder = TweezerThresholder(
        finder.images,
        new_site_rois,
        background_subtract=background_subtract,
        weights=finder.weight_functions(new_site_rois, background_subtract=background_subtract),
    )
thresholder.fit_gmms() # gmm stands for Gaussian mixture model
thresholder.overwrite_thresholds_to_yaml(folder)

fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(f'{folder}')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
multishot_analysis.tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')

fig.savefig(f'{folder}/tweezers_roi_detection.pdf')
