
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path

DO_ROI_DETECTION = False
DO_AVG_BACKGROUND = True

background_subtract = True
folder = select_data_directory()

if DO_ROI_DETECTION:
    finder = TweezerFinder.load_from_h5(folder)
    new_site_rois = finder.detect_rois_by_roi_number(roi_number=40, neighborhood_size=5, detection_threshold = 30)
    finder.overwrite_site_rois_to_yaml(new_site_rois, folder)
    finder.plot_sites(new_site_rois)

    # TODO: evaluate whether or not we actually should be subtracting the background for tweezers
    # TODO: Include survival rate if taking two shots
    thresholder = TweezerThresholder(
        finder.images,
        new_site_rois,
        background_subtract=background_subtract,
        weights=finder.weight_functions(new_site_rois, background_subtract=background_subtract),
    )


    multishot_analysis = TweezerMultishotAnalysis(folder)
else:
    tweezer_preproc = TweezerPreprocessor(
            load_type='h5',
            h5_path = next(folder.glob('20*0.h5')),
        )
    new_site_rois = tweezer_preproc.site_rois
    preproc_h5_path = Path(folder) / TweezerPreprocessor.PROCESSED_RESULTS_FNAME

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

fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(f'{folder}')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
if DO_ROI_DETECTION:
    multishot_analysis.tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])
else:
    tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')

fig.savefig(f'{folder}/tweezers_roi_detection.pdf')
