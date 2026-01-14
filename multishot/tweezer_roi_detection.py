import warnings

# workaround for following warning:
#     C:\Users\sslab\miniconda3\envs\labscript-env\lib\site-packages\sklearn\cluster\_kmeans.py:1419: UserWarning: 
#     KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. 
#     You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
# must be imported before numpy
# os.environ['OMP_NUM_THREADS'] = '1'  # doesn't work since this is run from Lyse rather than from a standalone process

import matplotlib.pyplot as plt
import numpy as np

from analysislib.common.analysis_config import kinetix_system
from analysislib.common.image import ROI
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory

background_subtract = True
USE_AVERAGED_BACKGROUND = True
folder = select_data_directory()

finder = TweezerFinder.load_from_h5(
    folder, use_averaged_background=USE_AVERAGED_BACKGROUND
)

restriction_roi = ROI(xmin=1100, xmax=1550, ymin=950, ymax=1050)  # None
# new_site_rois = finder.detect_rois_by_roi_number(
#     roi_number=50,
#     neighborhood_size=5,
#     roi_size=5,
#     detection_threshold=30,
#     restricted_ROI=restriction_roi,
# )

new_site_rois = finder.detect_rois_by_contours(
    roi_number=50,
    roi_size=5,
    restriction_roi=restriction_roi,
    blur_block=3,
    blur_width=1,
    block_size=11,
    relative_threshold=4,
    affine_transform=(lambda x: 4 * x + 40),
)
# finder.plot_sites(new_site_rois)
fig_contours = plt.figure(figsize=(10, 10), layout='constrained')
finder.plot_contour_site_detection(fig_contours)
fig_contours.suptitle(
    f'Tweezer site detection ({len(finder.images)} shots averaged)\n{str(folder)}',
)

thresholder = TweezerThresholder(
    finder.images,
    new_site_rois,
    background_subtract=background_subtract,
    weights=finder.weight_functions(
        new_site_rois, background_subtract=background_subtract
    ),
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    print('Fitting histograms...')
    thresholder.fit_gmms()

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

fig, ax = plt.subplots(layout='constrained')
thresholder.violinplot(ax)
fig.suptitle(folder)

fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(folder)
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
multishot_analysis.tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])

fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
multishot_analysis.tweezer_statistician.counts_scatterplot(ax=ax)
fig.suptitle(folder)

axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')

fig.savefig(folder / 'tweezers_roi_detection.pdf')
