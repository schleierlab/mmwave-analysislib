import matplotlib.pyplot as plt
import numpy as np

from analysislib.common.lab_constants import USERLIB_PATH
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.multishot.util import select_data_directory


SHOW_ROIS = True
background_subtract = True

folder = select_data_directory()

multishot_analysis = TweezerMultishotAnalysis(folder)
atom_roi = multishot_analysis.atom_roi
averaged_background = multishot_analysis.averaged_background()

fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
raw_img_color_kw = dict(
    cmap='viridis',
    # vmin=-10,
    # vmax=10,
)
im = ax.imshow(
    averaged_background[
        0:atom_roi.ymax - atom_roi.ymin,
        atom_roi.xmin:atom_roi.xmax,
    ],
    **raw_img_color_kw,
    )
fig.colorbar(im, ax=ax)

fig.savefig(folder / 'tweezers_average_background.pdf')

np.save(folder / 'avg_shot_bkg.npy', averaged_background)
np.save(USERLIB_PATH / 'analysislib/multishot/avg_shot_bkg.npy', averaged_background)
