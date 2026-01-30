import logging
import sys
import warnings
from pathlib import Path
from typing import cast

# workaround for following warning:
#     C:\Users\sslab\miniconda3\envs\labscript-env\lib\site-packages\sklearn\cluster\_kmeans.py:1419: UserWarning: 
#     KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. 
#     You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
# must be imported before numpy
# os.environ['OMP_NUM_THREADS'] = '1'  # doesn't work since this is run from Lyse rather than from a standalone process

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysislib.common.analysis_config import kinetix_system
from analysislib.common.image import ROI
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.lab_constants import ROI_CONFIG_PATH, USERLIB_PATH
from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_multishot import TweezerMultishotAnalyzer
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory


background_subtract = True
USE_AVERAGED_BACKGROUND = False
weighted_counts = True


def detect_rois(
        folder: Path,
        fig_summary: Figure,
):
    fig_summary.suptitle(str(folder))
    subfigs = fig_summary.subfigures(nrows=2, ncols=1, hspace=0.07, height_ratios=[2, 1])

    fig_contours_and_bg, fig_thresholding = subfigs[0].subfigures(nrows=1, ncols=2, wspace=0.07)
    fig_violin, fig_scatter = subfigs[1].subfigures(nrows=1, ncols=2, wspace=0.07, width_ratios=[3, 1])
    fig_contours, fig_bg = fig_contours_and_bg.subfigures(nrows=2, ncols=1, hspace=0.03, height_ratios=[3, 1])

    multishot_analyzer = TweezerMultishotAnalyzer(
        folder,
        use_averaged_background=USE_AVERAGED_BACKGROUND,
        background_subtract=False,
    )
    averaged_background = multishot_analyzer.mean_background()
    np.save(folder / 'avg_shot_bkg.npy', averaged_background)
    np.save(USERLIB_PATH / 'analysislib/multishot/avg_shot_bkg.npy', averaged_background)

    multishot_analyzer.background_subtraction(use_averaged_background=USE_AVERAGED_BACKGROUND)
    finder = TweezerFinder(multishot_analyzer.mean_image())

    new_site_rois = finder.detect_rois_by_contours(
        roi_number=50,
        roi_size=5,
        restriction_roi=ROI(xmin=1100, xmax=1550, ymin=950, ymax=1050),
        blur_block=3,
        blur_width=1,
        block_size=11,
        relative_threshold=4,
        affine_transform=(lambda x: 4 * x + 40),
    )
    finder.plot_contour_site_detection(fig_contours, ylims='full')
    fig_contours.suptitle(
        f'Tweezer site detection ({multishot_analyzer.n_shots} shots averaged)\n{str(folder)}',
    )

    if weighted_counts:
        weights = finder.weight_functions(
            new_site_rois, background_subtract=background_subtract
        )
    else:
        weights = 1

    thresholder = TweezerThresholder(
        multishot_analyzer.images(),
        new_site_rois,
        background_subtract=background_subtract,
        weights=weights,
    )

    # ignore KMeans memory leak warnings while fitting Gaussian mixture models
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        print('Fitting histograms...')
        thresholder.fit_gmms()

    # TODO: evaluate whether or not we actually should be subtracting the background for tweezers
    # TODO: Include survival rate if taking two shots

    # we use ImagePreprocessor because TweezerPreprocessor requires the existence of roi_config.yml already,
    # which would be circular (we're trying to generate that file here!)
    shots_h5s = folder.glob('20*.h5')
    processor = ImagePreprocessor(
        imaging_setups=[kinetix_system],
        load_type='h5',
        h5_path=next(shots_h5s),
    )
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
        output_path=ROI_CONFIG_PATH,
    )

    multishot_analyzer.analyze()

    ax_violin = fig_violin.subplots()
    thresholder.violinplot(ax_violin)

    axs = fig_thresholding.subplots(nrows=4, ncols=1, sharex=True)
    thresholder.plot_spreads(ax=axs[0])
    thresholder.plot_loading_rate(ax=axs[1])
    thresholder.plot_infidelity(ax=axs[2])
    multishot_analyzer.tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])

    axs[0].set_ylabel('Counts')
    axs[1].set_ylabel('Loading rate')
    axs[2].set_ylabel('Infidelity')
    axs[-1].set_xlabel('Tweezer index')

    ax_scatter = fig_scatter.subplots()
    multishot_analyzer.tweezer_statistician.counts_scatterplot(ax=ax_scatter)

    # hacky way to get atom roi...
    preproc: TweezerPreprocessor = next(iter(multishot_analyzer.preprocessors.values()))
    preproc.load_rois_threshs()
    atom_roi = preproc.atom_roi

    ax_avg = cast(Axes, fig_bg.subplots())
    im = ax_avg.imshow(
        averaged_background[
            0:atom_roi.ymax - atom_roi.ymin,
            atom_roi.xmin:atom_roi.xmax,
        ],
        cmap='viridis',
        # vmin=-10,
        # vmax=10,
    )
    fig_bg.colorbar(im, ax=ax_avg)
    ax_avg.set_title(f'Averaged background ({multishot_analyzer.n_shots} shots)')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    directory = select_data_directory()
    fig_summary = plt.figure(figsize=(12, 10), layout='constrained')
    detect_rois(directory, fig_summary)
