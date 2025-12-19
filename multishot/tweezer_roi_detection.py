import matplotlib.pyplot as plt

from analysislib.common.analysis_config import kinetix_system
from analysislib.common.image import ROI
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
import numpy as np

background_subtract = True
USE_AVERAGED_BACKGROUND = False
folder = select_data_directory()

finder = TweezerFinder.load_from_h5(
    folder, use_averaged_background=USE_AVERAGED_BACKGROUND
)

# ROI_restriction = ROI(xmin=1132, xmax=1510, ymin=997, ymax=1010)  # None
# new_site_rois = finder.detect_rois_by_roi_number(
#     roi_number=50,
#     neighborhood_size=5,
#     roi_size=5,
#     detection_threshold=30,
#     restricted_ROI=ROI_restriction,
# )

new_site_rois = finder.detect_rois_by_contours(
    roi_number=50,
    roi_size=5,
)
# finder.plot_sites(new_site_rois)
fig_contours = plt.figure(figsize=(10, 10), layout='constrained')
finder.plot_contour_site_detection(fig_contours)


thresholder = TweezerThresholder(
    finder.images,
    new_site_rois,
    background_subtract=background_subtract,
    weights=finder.weight_functions(
        new_site_rois, background_subtract=background_subtract
    ),
)
thresholder.fit_gmms()  # gmm stands for Gaussian mixture model

# TODO: evaluate whether or not we actually should be subtracting the background for tweezers
# TODO: Include survival rate if taking two shots

# we use ImagePreprocessor because TweezerPreprocessor requires the existence of roi_config.yml already,
# which would be circular (we're trying to generate that file here!)
shots_h5s = folder.glob('20*.h5')
processor = ImagePreprocessor(imaging_setup=kinetix_system, load_type='h5', h5_path=next(shots_h5s))
ymin, ymax = processor._load_ylims_from_globals()

padding = 50
if thresholder.thresholds is None:
    raise ValueError  # this should not happen since we have already found the thresholds first
TweezerPreprocessor.dump_to_yaml(
    new_site_rois,
    atom_roi=ROI(
        ymin=ymin,
        ymax=ymax,
        xmin=min(roi.xmin for roi in new_site_rois) - padding,
        xmax=max(roi.xmax for roi in new_site_rois) + padding,
    ),
    global_threshold=np.mean(thresholder.thresholds),
    site_thresholds=thresholder.thresholds,
    output_path=TweezerPreprocessor.ROI_CONFIG_PATH,
)

multishot_analysis = TweezerMultishotAnalysis(
    folder, use_averaged_background=USE_AVERAGED_BACKGROUND
)

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
