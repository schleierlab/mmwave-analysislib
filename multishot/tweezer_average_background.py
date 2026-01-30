import matplotlib.pyplot as plt
import numpy as np

from analysislib.common.lab_constants import USERLIB_PATH
from analysislib.common.tweezer_multishot import TweezerMultishotAnalyzer
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory


SHOW_ROIS = True
background_subtract = True

folder = select_data_directory()

multishot_analyzer = TweezerMultishotAnalyzer(folder)
averaged_background = multishot_analyzer.mean_background()

# hacky way to get atom roi...
preproc: TweezerPreprocessor = next(multishot_analyzer.preprocessors.values())
preproc.load_rois_threshs()
atom_roi = preproc.atom_roi

fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
im = ax.imshow(
    averaged_background[
        0:atom_roi.ymax - atom_roi.ymin,
        atom_roi.xmin:atom_roi.xmax,
    ],
    cmap='viridis',
    # vmin=-10,
    # vmax=10,
)
fig.colorbar(im, ax=ax)

fig.savefig(folder / 'tweezers_average_background.pdf')

np.save(folder / 'avg_shot_bkg.npy', averaged_background)
np.save(USERLIB_PATH / 'analysislib/multishot/avg_shot_bkg.npy', averaged_background)
