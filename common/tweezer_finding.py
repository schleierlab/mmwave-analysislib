from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from analysislib.common.image import ROI, Image
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

from typing import Union


class TweezerFinder:
    def __init__(self, images: Sequence[Image]):
        self.images = images
        self.averaged_image = Image.mean(images)

    def detect_rois_by_roi_number(
        self,
        roi_number: int,
        neighborhood_size: int = 5,
        detection_threshold: float = 25,
        roi_size: int = 5,
        search_step: float = 0.5,
    ):
        site_rois = self.detect_rois(
            neighborhood_size=neighborhood_size,
            detection_threshold=detection_threshold,
            roi_size=roi_size,
        )

        site_count_difference_init = len(site_rois) - roi_number
        if site_count_difference_init == 0:
            return site_rois

        current_threshold = detection_threshold
        init_difference_sign = +1 if site_count_difference_init > 0 else -1
        threshold_step = search_step * init_difference_sign
        while (len(site_rois) - roi_number) * init_difference_sign > 0:
            current_threshold += threshold_step
            print(f"Attempting threshold {current_threshold}")
            site_rois = self.detect_rois(
                neighborhood_size,
                current_threshold,
                roi_size,
            )

        print(f"Exactly {len(site_rois)} sites found")
        site_rois = self.remove_overlapping_rois(site_rois, min_distance=roi_size)

        return site_rois

    def remove_overlapping_rois(self, rois, min_distance):
        """
        Remove ROIs whose centers are closer than min_distance.
        Args:
            rois: list of ROI objects or tuples (x, y, ...)
            min_distance: minimum allowed distance between ROI centers
        Returns:
            Filtered list of ROIs
        """
        filtered = []
        centers = []
        for roi in rois:
            # Extract center coordinates depending on your ROI format
            if hasattr(roi, 'x') and hasattr(roi, 'y'):
                cx, cy = roi.x, roi.y
            else:
                cx, cy = roi[0], roi[1]  # adjust if your ROI is a tuple/list
            if all(np.hypot(cx - x0, cy - y0) >= min_distance for x0, y0 in centers):
                filtered.append(roi)
                centers.append((cx, cy))
        return filtered

    def detect_rois(self, neighborhood_size: int, detection_threshold: float, roi_size: int):
        return self.averaged_image.detect_site_rois(neighborhood_size, detection_threshold, roi_size)

    def weight_functions(self, rois: Sequence[ROI], background_subtract: bool = False):
        img = self.averaged_image if background_subtract else self.averaged_image.raw_image()
        return [
            img.roi_view(roi) / img.roi_sum(roi) * roi.pixel_area
            for roi in rois
        ]

    @classmethod
    def load_from_h5(cls, h5_path: Union[str, PathLike], use_averaged_background = False):
        sequence_dir = Path(h5_path)
        cls.folder = sequence_dir
        shots_h5s = sequence_dir.glob('20*.h5')
        print('Loading imagess...')
        images: list[Image] = []
        for shot in shots_h5s:
            print(shot)
            processor = TweezerPreprocessor(load_type='h5', h5_path=shot, use_averaged_background = use_averaged_background)
            images.append(processor.images[0])

        return cls(images)

    def overwrite_site_rois_to_yaml(self, new_site_rois: list[ROI], folder: str):
        """Overwrite the site ROIs in the roi_config.yml file, to be used by all subsequent
        TweezerPreprocessor instances.

        Parameters
        ----------
        new_site_rois : list[ROI]
            List of ROI objects for each site, to override the existing site ROIs in roi_config.yml
        folder : str
            Folder containing the sequence of h5 files of the current multishot analysis
        """
        sequence_dir = Path(folder)
        shots_h5s = sequence_dir.glob('20*.h5')
        processor = TweezerPreprocessor(load_type='h5', h5_path=next(shots_h5s))
        padding = 50

        xmin_lst = []
        xmax_lst = []
        for roi in new_site_rois:
            xmin_lst.append(roi.xmin)
            xmax_lst.append(roi.xmax)
        xmin_lst = np.array(xmin_lst)
        xmax_lst = np.array(xmax_lst)

        atom_roi = ROI(
            ymax = processor.atom_roi.ymax,
            ymin = processor.atom_roi.ymin,
            xmin = np.min(xmin_lst)- padding,
            xmax = np.max(xmax_lst) + padding
            )


        # The only reason we have to load the atom_roi this way, is because atom_roi_ylims is loaded
        # from the globals stored in the shot.h5 as tw_kinetix_roi_row.
        # TODO: If we could move the ylims to be stored in the roi_config.yml as the xlims are,
        # we could load the atom_roi to be copied in the same way that the threshold is copied below.

        roi_config_path = TweezerPreprocessor.ROI_CONFIG_PATH.parent / 'roi_config.yml'
        global_threshold, site_thresholds = TweezerPreprocessor._load_threshold_from_yaml(roi_config_path)
        output_path = TweezerPreprocessor.dump_to_yaml(new_site_rois,
                                                        atom_roi,
                                                        global_threshold,
                                                        site_thresholds,
                                                        roi_config_path
                                                        )
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
        text_kwargs = {
                    'color':'red',
                    'fontsize':'small',
                    }
        [ax.annotate(
            str(j), # The site index to display
            xy=(roi.xmin, roi.ymin - 5), # Position of the text
            **text_kwargs
            )
        # Iterate through sites, but only annotate if j is a multiple of 5
        for j, roi in enumerate(rois) if j % 5 == 0]

        fig.savefig(f'{self.folder}/tweezers_roi_detection_sites.pdf')


