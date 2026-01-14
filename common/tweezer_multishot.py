from os import PathLike
from pathlib import Path
from typing import Union

from tqdm import tqdm

from analysislib.common.image import Image
from analysislib.common.plot_config import PlotConfig
from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician


class TweezerMultishotAnalysis:
    """
    Class for analyzing the entire folder
    """

    def __init__(self, folder_path: Union[str, PathLike], use_averaged_background: bool = False):
        self.tweezer_statistician, bkg_image_lst, self.atom_roi, self.site_rois = self.analyze_the_folder(folder_path, use_averaged_background = use_averaged_background)
        self.bkg_image_lst = bkg_image_lst

    @classmethod
    def analyze_the_folder(cls, h5_path: Union[str, PathLike], use_averaged_background: bool = False):
        print('Loading images...')
        
        sequence_dir = Path(h5_path)
        shots_h5s = list(sequence_dir.glob('20*.h5'))
        pbar = tqdm(shots_h5s)

        bkg_image_lst: list[Image] = []
        
        for shot in pbar:
            pbar.set_description(str(shot.name))
            tweezer_preproc = TweezerPreprocessor(load_type='h5', h5_path=shot, use_averaged_background = use_averaged_background)
            atom_roi = tweezer_preproc.atom_roi
            site_rois = tweezer_preproc.site_rois
            # print(f"{atom_roi = }")
            processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=True)
            bkg_image_lst.append(tweezer_preproc.images[-1])

        tweezer_statistician = TweezerStatistician(
            preproc_h5_path=processed_results_fname,
            shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
            plot_config=PlotConfig(),
        )
        return tweezer_statistician, bkg_image_lst, atom_roi, site_rois

    def averaged_background(self):
        '''
        Returns the average background for the entire folder
        The average is calculated by averaging the background (last shot) of each image
        '''
        averaged_background = Image.mean(self.bkg_image_lst).background
        return averaged_background
