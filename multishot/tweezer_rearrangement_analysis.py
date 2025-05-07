
import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.multishot.util import select_data_directory
from analysislib.common.tweezer_multishot import TweezerMultishotAnalysis
from analysislib.common.tweezer_statistics import TweezerStatistician
from pathlib import Path
'''
Do tweezer multishot analysis without doing auto roi detection. Plot loading rate, histagrams, imaging fidelity...
Load data directly from 'tweezer_preprocess.h5' (this is the file generated after running tweezer_single_shot) to do multishot analysis.
'''
background_subtract = True
USE_AVERAGED_BACKGROUND = True
folder = select_data_directory()

tweezer_preproc = TweezerPreprocessor(
        load_type='h5',
        h5_path = next(folder.glob('20*0.h5')), # needed h5 file only to look at kinetix_roi_raw -- is there a better way to do this?
    )
new_site_rois = tweezer_preproc.site_rois
preproc_h5_path = Path(folder) / TweezerPreprocessor.PROCESSED_RESULTS_FNAME

target_array = tweezer_preproc.target_array
print(f'{target_array = }')



tweezer_statistician = TweezerStatistician(
                preproc_h5_path=preproc_h5_path,
            )

fig, axs = plt.subplots(nrows=3, ncols=1, layout='constrained')  # 3 plots in one row
fig.suptitle(f'{folder}')
tweezer_statistician.plot_rearrange_histagram(target_array, ax = axs[0])
tweezer_statistician.plot_rearrange_site_success_rate(target_array, ax = axs[1], plot_overlapping_histograms = True) # When set to False, plot only the histogram of all shots
tweezer_statistician.plot_site_loading_rates(ax = axs[2])
plt.tight_layout()
plt.show()

fig.savefig(f'{folder}/tweezers_rarrange_statistics.pdf')
