import matplotlib.pyplot as plt
import winsound
from typing import Literal

import numpy as np

from analysislib.common.tweezer_correlator import TweezerCorrelator
from analysislib.common.tweezer_preproc import TweezerPreprocessor

SHOW_ROIS = True
SHOW_INDEX = True  # site index will not show up if show_rois is set to False
USE_AVERAGED_BACKGROUND = True
FIT_TYPE_1D = None #'fringe_gauss_decay' #None
# do a curve fit at the final shot, set to None when don't do curve fit
# options: 'lorentzian', 'quadratic', 'fringe_exp_decay', 'fringe_gauss_decay', 'rabispec', None

SHOW_IMG_ONLY = False
EXACT_REARRANGEMENT = True
PLOT_PAIR_STATES = True
SHOW_HIST = False  # show histogram for survival rate
SAVE_DATA_CSV_FILE = False  # need to be False for 2d scans!
PARITY_SELECTION: Literal[0, 1, None] = 0
POST_SELECTION = None #"extremes" #None | "even" | "odd" | "extremes"
ATOM_PER_CLUSTER = 2

# Initialize analysis with background ROI and standard ROI loading
tweezer_preproc = TweezerPreprocessor(
    load_type='lyse', h5_path=None, use_averaged_background=USE_AVERAGED_BACKGROUND
)

fig = plt.figure(figsize=(12, 6), layout='constrained')
processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=True)
if SHOW_IMG_ONLY:
    tweezer_preproc.show_image(
        roi_patches=SHOW_ROIS, site_index=SHOW_INDEX, fig=fig, vmax=80
    )
else:
    subfigs = fig.subfigures(nrows=1, ncols=3, wspace=0.07)
    tweezer_preproc.show_image(
        roi_patches=SHOW_ROIS, site_index=SHOW_INDEX, fig=subfigs[0], vmax=80
    )

tweezer_correlator = TweezerCorrelator(
    preproc_h5_path=processed_results_fname,
    require_exact_rearrangement=True,
)


folder_path = tweezer_preproc.h5_path.parent

data_axs = subfigs[1].subplots(sharex=True, nrows=3)


# if not SHOW_IMG_ONLY:
    # if SAVE_DATA_CSV_FILE:
    #     indep_var, survival_rates, survival_rate_errs = (
    #         tweezer_stats.plot_survival_rate_1d_fig(
    #             fig=subfigs[1],
    #             fit_type=FIT_TYPE_1D,
    #             require_exact_rearrangement=EXACT_REARRANGEMENT,
    #             plot_pair_states=PLOT_PAIR_STATES,
    #             show_hist=SHOW_HIST,
    #         )
    #     )
    #     np.savetxt(
    #         folder_path / 'data.csv',
    #         [indep_var, survival_rates, survival_rate_errs],
    #         delimiter=',',
    #     )
    # else:
    #     tweezer_stats.plot_survival_rate(
    #         fig=subfigs[1],
    #         fit_type_1d=FIT_TYPE_1D,
    #         require_exact_rearrangement=EXACT_REARRANGEMENT,
    #         plot_pair_states=PLOT_PAIR_STATES,
    #         N = ATOM_PER_CLUSTER,
    #         postselect = POST_SELECTION,
    #         show_hist=SHOW_HIST,
    #     )

indep_var, _, _ = tweezer_correlator.plot_survival_rate_1d(ax=data_axs[0])
tweezer_correlator.plot_magnetization_pops(axs=data_axs[1])

plot_metric = 'parity'
if plot_metric == 'bitstrings':
    tweezer_correlator.plot_bitstring_heatmap(axs=data_axs[2])
elif plot_metric == 'parity':
    tweezer_correlator.plot_parity(ax=data_axs[2])
else:
    raise ValueError

tweezer_correlator._setup_xaxis(data_axs[-1], indep_var)
subfigs[1].align_labels()

tweezer_correlator.plot_tweezing_statistics(fig=subfigs[2], avg_loading_rate=False)


if tweezer_correlator.is_final_shot:
    figname = folder_path / 'tweezer_single_shot.pdf'
    fig.savefig(figname)

    # play a sound after a long run
    if tweezer_correlator.n_runs >= 50:
        notes = np.array([12, 7, 4, 0])  # do' sol mi do
        freqs = 440 * 2.0 ** ((notes - 9) / 12)
        for freq in freqs:
            winsound.Beep(int(freq), 300)
        winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

    # if tweezer_correlator.polymer_length > 1:
    #     fig_corr.savefig(folder_path / 'tweezer_polymer_analysis.pdf')
# tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
