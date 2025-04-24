
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path
import numpy as np

SHOW_ROIS = True
background_subtract = True
folder = select_data_directory()

multishot_analysis = TweezerMultishotAnalysis(folder)
average_background = multishot_analysis.average_background

fig = plt.figure(layout='constrained', figsize=(10, 4))
multishot_analysis.tweezer_preproc.show_image(roi_patches=SHOW_ROIS, fig=fig, vmax=10)

fig.savefig(f'{folder}/tweezers_average_background.pdf')

np.save(f'{folder}/avg_shot_bkg.npy', average_background)
