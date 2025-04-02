from collections.abc import Sequence
from pathlib import Path

from analysislib.common.image import ROI, Image
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np


class TweezerFinder:
    def __init__(self, images: Sequence[Image]):
        self.images = images
        self.averaged_image = Image.mean(images)

    def detect_rois(self, neighborhood_size: int, detection_threshold: float, roi_size: int):
        return self.averaged_image.detect_site_rois(neighborhood_size, detection_threshold, roi_size)

    def weight_functions(self, rois: Sequence[ROI], background_subtract: bool = False):
        img = self.averaged_image if background_subtract else self.averaged_image.raw_image()
        return [
            img.roi_view(roi) / img.roi_sum(roi) * roi.pixel_area
            for roi in rois
        ]

    @classmethod
    def load_from_h5(cls, h5_path: str):
        sequence_dir = Path(h5_path)
        shots_h5s = sequence_dir.glob('20*.h5')

        print('Loading imagess...')
        images: list[Image] = []
        for shot in shots_h5s:
            print(shot)
            processor = TweezerPreprocessor(load_type='h5', h5_path=shot)
            images.append(processor.images[0])

        return cls(images)

    def overwrite_site_rois_to_yaml(self, new_site_rois: list[ROI], folder: str):
        """Overwrite the site ROIs in the roi_config.yml file, to be used by all subsequent
        TweezerPreprocessor instances.
        
        Parameters
        ----------
        new_site_rois : list[ROI]
            List of ROI objects for each site
        """
        sequence_dir = Path(folder)
        shots_h5s = sequence_dir.glob('20*.h5')
        processor = TweezerPreprocessor(load_type='h5', h5_path=next(shots_h5s))
        atom_roi = processor.atom_roi
        threshold = processor.threshold
        output_path = TweezerPreprocessor.ROI_CONFIG_PATH.parent / 'roi_test.yml'
        output_path = TweezerPreprocessor.dump_to_yaml(new_site_rois, atom_roi, threshold, output_path)
        print(f'Site ROIs dumped to {output_path}')

    def plot_sites(self, rois: Sequence[ROI]):
        fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')

        rois_bbox = ROI.bounding_box(rois)
        padding = 50
        atom_roi_xlims = [rois_bbox.xmin - padding, rois_bbox.xmax + padding]

        fig.suptitle(f'Tweezer site detection ({len(self.images)} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=np.max(self.averaged_image.subtracted_array),
        )

        full_ylims = (
            self.averaged_image.yshift,
            self.averaged_image.yshift + self.averaged_image.height,
        )

        im = self.averaged_image.imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=ax,
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=ax)

        patchs = tuple(roi.patch(edgecolor='yellow') for roi in rois)
        collection = PatchCollection(patchs, match_original=True)
        ax.add_collection(collection)


