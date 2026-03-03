import sys
from typing import Literal

from matplotlib import pyplot as plt

from analysislib.common.tweezer_correlator import TweezerCorrelator
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.singleshot.util import load_h5_path

PARITY_SELECTION: Literal[0, 1, None] = 0
EXACT_REARRANGEMENT = True

h5path = load_h5_path()
preproc_h5_path = h5path.with_name(TweezerPreprocessor.PROCESSED_RESULTS_FNAME)
tweezer_correlator = TweezerCorrelator(
    preproc_h5_path=preproc_h5_path,
    require_exact_rearrangement=EXACT_REARRANGEMENT,
    parity_selection=PARITY_SELECTION,
)

if tweezer_correlator.polymer_length == 1:
    sys.exit(0)
elif tweezer_correlator.polymer_length == 2:
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    tweezer_correlator.plot_bitstring_populations(ax)
else:
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 12), layout='constrained')
    tweezer_correlator.plot_magnetization_pops(axs[0])
    tweezer_correlator.plot_local_magnetization(axs[1])
    tweezer_correlator.plot_distance_averaged_correlation(axs[2])

if PARITY_SELECTION is None:
    parity_selection_str = 'parity post-selection: OFF'
else:
    parity_selection_str = f'parity post-selection: {PARITY_SELECTION} mod 2'

exact_rearr_str = f'perfect rearrangement post-sel: {EXACT_REARRANGEMENT}'

fig.suptitle('\n'.join([
    str(tweezer_correlator.folder_path),
    parity_selection_str,
    exact_rearr_str,
]))
