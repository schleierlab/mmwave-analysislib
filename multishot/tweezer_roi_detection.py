from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.tweezer_histograms import TweezerThresholder
from analysislib.common.tweezer_preproc import TweezerPreprocessor


# show dialog box and return the path
while True:
    try:
        folder = askdirectory(title='Select data directory for tweezer site detection')
    except Exception as e:
        raise e
    break

finder = TweezerFinder.load_from_h5(folder)
new_site_rois = finder.detect_rois(
    neighborhood_size=5,
    detection_threshold=25,
    roi_size=5,
)
print(new_site_rois)
print(folder)
finder.overwrite_site_rois_to_yaml(new_site_rois, folder)
finder.plot_sites(new_site_rois)

background_subtract = False
thresholder = TweezerThresholder(
    finder.images,
    new_site_rois,
    background_subtract=background_subtract,
    weights=finder.weight_functions(new_site_rois, background_subtract=background_subtract),
)
thresholder.fit_gmms()

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, layout='constrained')
thresholder.plot_spreads(ax=axs[0])
thresholder.plot_loading_rate(ax=axs[1])
thresholder.plot_infidelity(ax=axs[2])
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Loading rate')
axs[2].set_ylabel('Infidelity')
axs[-1].set_xlabel('Tweezer index')
