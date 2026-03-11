import matplotlib.pyplot as plt
import numpy as np

from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_correlator import TweezerCorrelatorVibed
from analysislib.multishot.util import select_data_directory

shot_mask = np.array([0,1])
# shot_mask = np.array([[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]])
# shot_mask = np.array([[0,1,2], [3,4,5], [6,7,8] ,[9,10,11]])
# shot_mask = np.array([[0,1,2,3], [4,5,6,7], [8,9,10,11]])
# shot_mask = np.array([0,1,2,3,4,5])

background_subtract = True
USE_AVERAGED_BACKGROUND = False
folder = select_data_directory()

tweezer_preproc = TweezerPreprocessor(
    load_type='h5',
    h5_path = next(folder.glob('20*0.h5')), # needed h5 file only to look at kinetix_roi_raw -- is there a better way to do this?
)
site_rois = tweezer_preproc.site_rois
preproc_h5_path = folder / TweezerPreprocessor.PROCESSED_RESULTS_FNAME

tweezer_correlator = TweezerCorrelatorVibed(
    preproc_h5_path=preproc_h5_path,
    target_sites=tweezer_preproc.target_array,
)

#Ask Tony about re-thresholding. Is there an easy way to re-process the shots with new thresholds
#Similarly, look into making a separate threshold for the rearranged atoms
thresholder = TweezerThresholder(
    None,
    site_rois,
    background_subtract=background_subtract,
    weights=None,
    processed_results_fname=preproc_h5_path, 
)

thresholder.fit_gmms() # gmm stands for Gaussian mixture model
thresholder.overwrite_thresholds_to_yaml(folder)

fig_hist, ax_hist = plt.subplots(layout='constrained')
thresholder.plot_histogram(ax_hist)
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, layout='constrained')
fig.suptitle(f'{folder}')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
tweezer_correlator.plot_survival_rate_by_site(ax=axs[3])

# fig = plt.figure(figsize=(10, 6))
# tweezer_correlator.plot_survival_rate_1d_fig(fig, plot_pair_states = True)


#Correlation stuff
# fig = plt.figure(figsize=(10, 6))
# tweezer_correlator.plot_bitstring_populations_heatmap(
#         fig = fig,
#         shot_mask = shot_mask,
#         require_exact_rearrangement = True,
#         average_shot_masks = False,
# )

fig = plt.figure(figsize=(10, 6))
tweezer_correlator.plot_bitstring_populations_curves(
        fig = fig,
        shot_mask = shot_mask,
        require_exact_rearrangement = True,
        average_shot_masks = True,
        group_by_magnetization = False,
)
