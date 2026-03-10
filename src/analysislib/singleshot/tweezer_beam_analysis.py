import matplotlib.pyplot as plt
import numpy as np
import uncertainties
import uncertainties.unumpy as unp

from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.analysis_config import manta_tweezer_system
from analysislib.common.image import Image


preprocessor = ImagePreprocessor(imaging_setups=[manta_tweezer_system], load_type='lyse')

# make tweezerfinder
image = Image(preprocessor.exposures_dict[manta_tweezer_system][0])
finder = TweezerFinder(image)
site_number = len(preprocessor.parameters['TW_x_freqs'])

site_rois = finder.detect_rois_by_contours(
    roi_number=site_number,
    roi_size=15,
    blur_block=5,
    blur_width=2,
    block_size=31,
    relative_threshold=6,
)
fig_contours = plt.figure(figsize=(10, 10), layout='constrained')
finder.plot_contour_site_detection(fig_contours, ylims='sites', label_displacement=(-12, -12))
# fig_contours.suptitle(str(folder))

# fit site rois
fitted_upopts = np.array([
    uncertainties.correlated_values(
        *image.roi_fit_gaussian2d(roi, isotropic=True),
    )
    for roi in site_rois
])

fig, axs = plt.subplots(nrows=3, layout='constrained', sharex=True)
x_uopt, y_uopt, width_uopt, amp_uopt, offset_uopt = fitted_upopts.T
# integrated_counts_u = 2 * pi * width_uopt**2 * amp_uopt

axs[0].errorbar(
    np.arange(site_number),
    unp.nominal_values(amp_uopt),
    unp.std_devs(amp_uopt),
)
axs[0].set_ylabel('Peak counts (fitted)')

waists_um_u = 2 * 1e+6 * width_uopt * manta_tweezer_system.atom_plane_pixel_size
axs[1].errorbar(
    np.arange(site_number),
    unp.nominal_values(waists_um_u),
    unp.std_devs(waists_um_u),
)

mean_waist_um_u = np.mean(waists_um_u)
axs[1].axhline(
    mean_waist_um_u.n,
    label=f'Mean waist: {mean_waist_um_u:S} $\mu$m',
)
axs[1].set_ylabel(R'Atom plane waist $w$ ($\mu$m)')

distances = unp.sqrt((x_uopt[1:] - x_uopt[:-1])**2 + (y_uopt[1:] - y_uopt[:-1])**2)
distances_um_u = 1e+6 * distances * manta_tweezer_system.atom_plane_pixel_size

axs[2].errorbar(
    np.arange(site_number-1) + 0.5,
    unp.nominal_values(distances_um_u),
    unp.std_devs(distances_um_u),   
)

mean_distance_um_u = np.mean(distances_um_u)
axs[2].axhline(
    mean_distance_um_u.n,
    label=f'Mean distance: {mean_distance_um_u:S} $\mu$m',
)
axs[2].set_ylabel(R'Atom plane neighbor distance $d$ ($\mu$m)')

axs[-1].set_xlabel('Tweezer site')
for ax in axs[1:]:
    ax.legend()

fig.align_labels()
