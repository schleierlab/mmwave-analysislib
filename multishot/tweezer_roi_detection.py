from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from analysislib.common.analysis_config import kinetix_system
from analysislib.common.image import ROI, Image
from analysislib.common.tweezer_analysis import TweezerPreprocessor


# show dialog box and return the path
while True:
    try:
        folder = askdirectory(title='Select data directory for tweezer site detection')
    except Exception as e:
        raise e
    break

sequence_dir = Path(folder)
shots_h5s = sequence_dir.glob('20*.h5')

print('Loading images ...')
images: list[Image] = []
for shot in shots_h5s:
    print(shot)
    processor = TweezerPreprocessor(load_type='h5', h5_path=shot)
    images.append(processor.images[0])

averaged_image = Image.mean(images)
full_ylims = [processor.atom_roi.ymin, processor.atom_roi.ymax]
site_rois = averaged_image.detect_site_rois(neighborhood_size=5, detection_threshold=25, roi_size=7)

rois_bbox = ROI.bounding_box(site_rois)
padding = 50
atom_roi_xlims = [rois_bbox.xmin - padding, rois_bbox.xmax + padding]

fig, ax = plt.subplots(layout='constrained')
fig.suptitle(f'Tweezer site detection ({len(images)} shots averaged)')
raw_img_color_kw = dict(
    cmap='viridis',
    vmin=0,
    vmax=np.max(averaged_image.subtracted_array),
)
im = averaged_image.imshow_view(
    ROI.from_roi_xy(atom_roi_xlims, full_ylims),
    ax=ax,
    **raw_img_color_kw,
)
fig.colorbar(im, ax=ax)

patchs = tuple(roi.patch(edgecolor='yellow') for roi in site_rois)
collection = PatchCollection(patchs, match_original=True)
ax.add_collection(collection)

root = Tk()
root.destroy()
