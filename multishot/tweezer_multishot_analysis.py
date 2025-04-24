
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
folder = select_data_directory()

tweezer_preproc = TweezerPreprocessor(
        load_type='h5',
        h5_path = next(folder.glob('20*0.h5')), # needed h5 file only to look at kinetix_roi_raw -- is there a better way?
    )
new_site_rois = tweezer_preproc.site_rois
preproc_h5_path = Path(folder) / TweezerPreprocessor.PROCESSED_RESULTS_FNAME

tweezer_statistician = TweezerStatistician(
                preproc_h5_path=preproc_h5_path,
            )

thresholder = TweezerThresholder(
    None,
    new_site_rois,
    background_subtract=background_subtract,
    weights=None,
    processed_results_fname=preproc_h5_path
)

thresholder.fit_gmms() # gmm stands for Gaussian mixture model
thresholder.overwrite_thresholds_to_yaml(folder)

# TODO: add function to plot averaged tweezer image
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(f'{folder}')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
tweezer_statistician.plot_survival_rate_by_site(ax=axs[3])
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')

fig.savefig(f'{folder}/tweezers_roi_detection.pdf')
