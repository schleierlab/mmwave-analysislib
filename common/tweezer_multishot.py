from os import PathLike
from pathlib import Path

from analysislib.common.tweezer_preproc import TweezerPreprocessor
from analysislib.common.tweezer_statistics import TweezerStatistician
from analysislib.common.plot_config import PlotConfig
from .image import Image

from typing import Union



class TweezerMultishotAnalysis():
    """
    Class for analyzing the entire folder
    """

    def __init__(self, folder_path: Union[str, PathLike]):
        self.tweezer_statistician, self.tweezer_preproc = self.analyze_the_folder(folder_path)
        self.average_background = self.average_background(self.tweezer_preproc)

    @classmethod
    def analyze_the_folder(cls, h5_path: Union[str, PathLike]):
        sequence_dir = Path(h5_path)
        shots_h5s = sequence_dir.glob('20*.h5')

        print('Loading imagess...')
        for shot in shots_h5s:
            print(shot)
            tweezer_preproc = TweezerPreprocessor(load_type='h5', h5_path=shot)
            processed_results_fname = tweezer_preproc.process_shot(use_global_threshold=True)

        tweezer_statistician = TweezerStatistician(
            preproc_h5_path=processed_results_fname,
            shot_h5_path=tweezer_preproc.h5_path, # Used only for MLOOP
            plot_config=PlotConfig(),
        )
        return tweezer_statistician, tweezer_preproc


    def average_background(self, tweezer_preproc):
        '''
        Returns the average background for the entire folder
        The average is calculated by averaging the background (last shot) of each image
        '''

        average_image = Image.mean(tweezer_preproc.images)
        average_background = average_image.background

        return average_background

