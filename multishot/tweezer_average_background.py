
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SHOW_ROIS = True
background_subtract = True

average_background_overwrite_path = Path(r'X:\userlib\analysislib\multishot')
folder = select_data_directory()

multishot_analysis = TweezerMultishotAnalysis(folder)
atom_roi = multishot_analysis.atom_roi
averaged_background = multishot_analysis.averaged_background

fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
raw_img_color_kw = dict(
    cmap='viridis',
    # vmin=-10,
    # vmax=10,
)
im = ax.imshow(
    averaged_background[
        0:atom_roi.ymax - atom_roi.ymin,
        atom_roi.xmin:atom_roi.xmax
        ],
    **raw_img_color_kw,
    )
fig.colorbar(im, ax=ax)

fig.savefig(f'{folder}/tweezers_average_background.pdf')

np.save(f'{folder}/avg_shot_bkg.npy', averaged_background)
np.save(f'{average_background_overwrite_path}/avg_shot_bkg.npy', averaged_background)
