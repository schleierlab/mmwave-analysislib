from datetime import datetime

import matplotlib.pyplot as plt
import addcopyfighandler
import winsound

import numpy as np

from analysislib.common.tweezer_correlator import TweezerCorrelator
from analysislib.common.tweezer_preproc import TweezerPreprocessor

SHOW_ROIS = True
SHOW_INDEX = True  # site index will not show up if show_rois is set to False
USE_AVERAGED_BACKGROUND = True
FIT_TYPE_1D = 'fringe_exp_decay' #'fringe_gauss_decay' #None
# do a curve fit at the final shot, set to None when don't do curve fit
# options: 'lorentzian', 'quadratic', 'fringe_exp_decay', 'fringe_gauss_decay', 'rabispec', None

SHOW_IMG_ONLY = False
PLOT_METRIC = 'parity' # options: 'bitstrings', 'parity'
PLOT_EVERY = 200

# 0, 1, or None
PARITY_SELECTION = None

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


folder_path = tweezer_preproc.h5_path.parent
subfigs[1].suptitle(f'Parity selection: {PARITY_SELECTION} mod 2')

def data_plots(correlator: TweezerCorrelator):
    data_axs = subfigs[1].subplots(sharex=True, nrows=3)

    indep_var, _, _ = correlator.plot_survival_rate_1d(ax=data_axs[0])
    correlator.plot_magnetization_pops(axs=data_axs[1])

    if PLOT_METRIC == 'bitstrings':
        correlator.plot_bitstring_heatmap(axs=data_axs[2])
    elif PLOT_METRIC == 'parity':
        correlator.plot_parity(ax=data_axs[2])
    else:
        raise ValueError

    correlator._setup_xaxis(data_axs[-1], indep_var)
    subfigs[1].align_labels()

    correlator.plot_tweezing_statistics(fig=subfigs[2], avg_loading_rate=False)


    if correlator.is_final_shot:
        figname = folder_path / 'tweezer_single_shot.pdf'
        fig.savefig(figname)
        addcopyfighandler.copyfig(fig)

        # play a sound after a long run
        if correlator.n_runs >= 50:
            notes = np.array([12, 7, 4, 0])  # do' sol mi do
            freqs = 440 * 2.0 ** ((notes - 9) / 12)
            for freq in freqs:
                winsound.Beep(int(freq), 300)
            winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

        # if correlator.polymer_length > 1:
        #     fig_corr.savefig(folder_path / 'tweezer_polymer_analysis.pdf')
    # tweezer_statistician.plot_survival_rate_by_site(fig=subfigs[1])


if tweezer_preproc.run_number % PLOT_EVERY == 0 or tweezer_preproc.run_number + 1 == tweezer_preproc.n_runs:
    start = datetime.now()
    tweezer_correlator = TweezerCorrelator(
        preproc_h5_path=processed_results_fname,
        require_exact_rearrangement=True,
        parity_selection=PARITY_SELECTION,
    )

    init_end = datetime.now()

    data_plots(tweezer_correlator)

    plot_end = datetime.now()

    print(f'{init_end-start=}, {plot_end-init_end=}')
