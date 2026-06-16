import matplotlib.pyplot as plt
import addcopyfighandler
import winsound

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
import numpy as np

SHOW_ROIS = True
SHOW_INDEX = True  # site index will not show up if show_rois is set to False
USE_AVERAGED_BACKGROUND = True
FIT_TYPE_1D = None#'quadratic'# "fringe_gauss_decay"#'sinusoidal' #"fringe_exp_decay"
# do a curve fit at the final shot, set to None when don't do curve fit
# options: 'lorentzian', 'quadratic', 'fringe_exp_decay', 'fringe_gauss_decay', 'rabispec', 'exp_decay', 'sinusoidal', None

SHOW_IMG_ONLY = False
EXACT_REARRANGEMENT = False
REQUIRE_NO_EXTRA_ATOMS = True
PLOT_PAIR_STATES = False
SHOW_HIST = False  # show histogram for survival rate
SAVE_DATA_CSV_FILE = False  # need to be False for 2d scans!
ATOM_PER_CLUSTER = 1

USE_GLOBAL_THRESHOLD = True
USE_REARRANGEMENT_THRESHOLD = False

# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse', h5_path=None, use_averaged_background=USE_AVERAGED_BACKGROUND
)

fig = plt.figure(figsize=(12, 6), layout='constrained')

processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=USE_GLOBAL_THRESHOLD, use_rearrangement_threshold=USE_REARRANGEMENT_THRESHOLD)
if SHOW_IMG_ONLY:
    tweezer_preproc.show_image(
        roi_patches=SHOW_ROIS, site_index=SHOW_INDEX, fig=fig, vmax=80
    )
else:
    subfigs = fig.subfigures(nrows=1, ncols=3, wspace=0.07)
    tweezer_preproc.show_image(
        roi_patches=SHOW_ROIS, site_index=SHOW_INDEX, fig=subfigs[0], vmax=80
    )

tweezer_stats = TweezerStatistician(
    preproc_h5_path=processed_results_fname,
)


folder_path = tweezer_preproc.h5_path.parent
if not SHOW_IMG_ONLY:
    if SAVE_DATA_CSV_FILE:
        indep_var, survival_rates, survival_rate_errs = (
            tweezer_stats.plot_survival_rate_1d_fig(
                fig=subfigs[1],
                fit_type=FIT_TYPE_1D,
                require_exact_rearrangement=EXACT_REARRANGEMENT,
                require_no_extra_atoms=REQUIRE_NO_EXTRA_ATOMS,
                show_hist=SHOW_HIST,
            )
        )
        np.savetxt(
            folder_path / 'data.csv',
            [indep_var, survival_rates, survival_rate_errs],
            delimiter=',',
        )
    else:
        tweezer_stats.plot_survival_rate(
            fig=subfigs[1],
            fit_type_1d=FIT_TYPE_1D,
            require_exact_rearrangement=EXACT_REARRANGEMENT,
            require_no_extra_atoms=REQUIRE_NO_EXTRA_ATOMS,
            show_hist=SHOW_HIST,
        )

    tweezer_stats.plot_tweezing_statistics(fig=subfigs[2], avg_loading_rate=False)

    # TODO: this function right now doesn't work with 2d parameter scan

if tweezer_stats.is_final_shot:
    figname = folder_path / 'tweezer_single_shot.pdf'
    fig.savefig(figname)
    addcopyfighandler.copyfig(fig)

    # play a sound after a long run
    if tweezer_stats.n_runs >= 50:
        notes = np.array([12, 7, 4, 0])  # do' sol mi do
        freqs = 440 * 2.0 ** ((notes - 9) / 12)
        for freq in freqs:
            winsound.Beep(int(freq), 300)
        winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
