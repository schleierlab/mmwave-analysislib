import matplotlib.pyplot as plt
import os
import winsound
from typing import Literal

from analysislib.common.tweezer_correlator import TweezerCorrelator
from analysislib.common.tweezer_preproc import TweezerPreprocessor
import numpy as np

SHOW_ROIS = True
SHOW_INDEX = True  # site index will not show up if show_rois is set to False
USE_AVERAGED_BACKGROUND = True
FIT_TYPE_1D = None
# do a curve fit at the final shot, set to None when don't do curve fit
# options: 'lorentzian', 'quadratic', 'decay_exp', 'decay_gauss', 'rabispec', None

SHOW_IMG_ONLY = False
EXACT_REARRANGEMENT = True
PLOT_PAIR_STATES = False
SHOW_HIST = False  # show histogram for survival rate
SAVE_DATA_CSV_FILE = False  # need to be False for 2d scans!
PARITY_SELECTION: Literal[0, 1, None] = 0

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
    require_exact_rearrangement=EXACT_REARRANGEMENT,
    parity_selection=PARITY_SELECTION,
)


folder_path = os.path.dirname(tweezer_preproc.h5_path)
if not SHOW_IMG_ONLY:
    if SAVE_DATA_CSV_FILE:
        indep_var, survival_rates, survival_rate_errs = (
            tweezer_correlator.plot_survival_rate_1d(
                fig=subfigs[1],
                fit_type=FIT_TYPE_1D,
                require_exact_rearrangement=EXACT_REARRANGEMENT,
                plot_pair_states=PLOT_PAIR_STATES,
                show_hist=SHOW_HIST,
            )
        )
        np.savetxt(
            folder_path + '/data.csv',
            [indep_var, survival_rates, survival_rate_errs],
            delimiter=',',
        )
    else:
        tweezer_correlator.plot_survival_rate(
            fig=subfigs[1],
            fit_type_1d=FIT_TYPE_1D,
            require_exact_rearrangement=EXACT_REARRANGEMENT,
            plot_pair_states=PLOT_PAIR_STATES,
            show_hist=SHOW_HIST,
        )

    tweezer_correlator.plot_tweezing_statistics(fig=subfigs[2], avg_loading_rate=False)

    # TODO: this function right now doesn't work with 2d parameter scan


def correlation_plot(correlator):
    if correlator.polymer_length == 1:
        raise ValueError
    elif correlator.polymer_length == 2:
        fig_corr, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        correlator.plot_bitstring_populations(ax)
    else:
        fig_corr, axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 12), layout='constrained')
        correlator.plot_magnetization_pops(axs[0])
        correlator.plot_local_magnetization(axs[1])
        correlator.plot_distance_averaged_correlation(axs[2])

    if PARITY_SELECTION is None:
        parity_selection_str = 'parity post-selection: OFF'
    else:
        parity_selection_str = f'parity post-selection: {PARITY_SELECTION} mod 2'

    exact_rearr_str = f'perfect rearrangement post-sel: {EXACT_REARRANGEMENT}'

    fig_corr.suptitle('\n'.join([
        str(correlator.folder_path),
        parity_selection_str,
        exact_rearr_str,
    ]))

    return fig_corr


if tweezer_correlator.polymer_length > 1:
    fig_corr = correlation_plot(tweezer_correlator)

if tweezer_correlator.is_final_shot:
    figname = folder_path + '/tweezer_single_shot.pdf'
    fig.savefig(figname)

    # play a sound after a long run
    if tweezer_correlator.n_runs >= 50:
        notes = np.array([12, 7, 4, 0])  # do' sol mi do
        freqs = 440 * 2.0 ** ((notes - 9) / 12)
        for freq in freqs:
            winsound.Beep(int(freq), 300)
        winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

# tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])
