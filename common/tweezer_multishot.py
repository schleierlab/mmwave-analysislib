from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from analysislib.common.image import Image, ROI
from analysislib.common.plot_config import PlotConfig
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.typing import StrPath


def _requires_bg(func):
    def decorated(self: TweezerMultishotAnalyzer, *args, **kwargs):
        if not self._backgrounds_subtracted:
            raise ValueError('Method requires background subtraction to be run first.')
        return func(self, *args, **kwargs)
    
    return decorated


class TweezerMultishotAnalyzer:
    _backgrounds_subtracted: bool = False
    path: Path
    preprocessors: dict[Path, TweezerPreprocessor]

    def __init__(
            self,
            path: StrPath,
            use_averaged_background: Optional[bool] = None,
            background_subtract: bool = True,
    ):
        self.path = Path(path)
        self._load_exposures()
        if background_subtract:
            if use_averaged_background is None:
                raise ValueError('If doing background subtraction, must specify whether to use averaged background.')
            self.background_subtraction(use_averaged_background)

    def _load_exposures(self) -> None:
        """Load the raw exposure data."""
        print('Loading exposures...')

        shots_h5s = list(self.path.glob('20*.h5'))
        pbar = tqdm(shots_h5s)

        preprocessors: dict[Path, TweezerPreprocessor] = {}
        for shot in pbar:
            pbar.set_description(str(shot.name))
            preproc = TweezerPreprocessor(
                load_type='h5',
                h5_path=shot,
                initialize=False,
            )
            preprocessors[shot] = preproc

        self.preprocessors = preprocessors

    @property
    def n_shots(self) -> int:
        return len(self.preprocessors)

    def mean_background(self) -> NDArray:
        return np.mean([preproc.exposures[-1] for preproc in self.preprocessors.values()], axis=0)

    def background_subtraction(self, use_averaged_background: bool):
        for preproc in self.preprocessors.values():
            preproc.background_subtraction(use_averaged_background)
        self._backgrounds_subtracted = True

    @_requires_bg
    def images(self, index: int = 0) -> list[Image]:
        return [preproc.images[index] for preproc in self.preprocessors.values()]

    @_requires_bg
    def mean_image(self, index: int = 0) -> Image:
        return Image.mean(self.images(index))

    @_requires_bg
    def analyze(self):
        print('Analyzing shots...')
        
        pbar = tqdm(self.preprocessors)
        for shot in pbar:
            pbar.set_description(str(shot.name))
            tweezer_preproc = self.preprocessors[shot]
            tweezer_preproc.load_rois_threshs()
            processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=True)

        self.tweezer_statistician = TweezerStatistician(
            preproc_h5_path=processed_results_fname,
            shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
            plot_config=PlotConfig(),
        )
    
    # not sure if this belongs here; putting it here to limit scope of present diff
    @_requires_bg
    def plot_averaged_images(self, rois: Sequence[ROI]):
        '''
        Plot the average of the 1st and 2nd images taken from each shot seperately
        Doesn't include rois for now
        '''
        fig, axs = plt.subplots(nrows=2, ncols=1, layout='constrained')

        rois_bbox = ROI.bounding_box(rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)

        mean_images = [self.mean_image(index=i) for i in range(2)]

        fig.suptitle(f'Tweezer site detection ({self.n_shots} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=max(np.max(mean_image.subtracted_array) for mean_image in mean_images),
        )

        yshift = mean_images[0].yshift
        full_ylims = (yshift, yshift + mean_images[0].height)

        im = self.mean_image(index=0).imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=axs[0],
            **raw_img_color_kw,
        )
        im = self.mean_image(index=1).imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=axs[1],
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=axs)
        fig.savefig(self.path / 'tweezers_averaged_images.pdf')

    # not sure if this belongs here; putting it here to limit scope of present diff
    # TODO should be merged with above function
    @_requires_bg
    def plot_sites(self, rois: Sequence[ROI]):
        '''
        Plot the average of the 1st image taken from each shot
        Include rois
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')

        rois_bbox = ROI.bounding_box(rois)
        padding = 50
        atom_roi_xlims = (rois_bbox.xmin - padding, rois_bbox.xmax + padding)

        mean_images = [self.mean_image(index=i) for i in range(1)]

        fig.suptitle(f'Tweezer site detection ({self.n_shots} shots averaged)')
        raw_img_color_kw = dict(
            cmap='viridis',
            vmin=0,
            vmax=max(np.max(mean_image.subtracted_array) for mean_image in mean_images),
        )

        yshift = mean_images[0].yshift
        full_ylims = (yshift, yshift + mean_images[0].height)

        im = mean_images[0].imshow_view(
            ROI.from_roi_xy(atom_roi_xlims, full_ylims),
            ax=ax,
            **raw_img_color_kw,
        )
        fig.colorbar(im, ax=ax)

        ROI.plot_rois(
            ax,
            rois,
            label_sites=5,
        )

        return fig
