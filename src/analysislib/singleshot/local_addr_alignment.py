import logging
import sys
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysislib.common.analysis_config import (
    ImagingSystem,
    manta_local_addr_align_system,
    manta_tweezer_system,
)
from analysislib.common.beam_image_preproc import BeamImagePreprocessor
from analysislib.common.beam_image_preproc import BeamDetectionConfig
from analysislib.common.typing import StrPath


logger = logging.getLogger(__name__)


la_align_detection_configs: dict[ImagingSystem, Union[BeamDetectionConfig, tuple[BeamDetectionConfig, ...]]] = {
    manta_tweezer_system: BeamDetectionConfig(
        scaling_factor=16,
        roi_size=21,
        blur_block=3,
        blur_width=1,
        block_size=11,
        relative_threshold=+2,
    ),
    manta_local_addr_align_system: (
        BeamDetectionConfig(
            scaling_factor=8,
            roi_size=300,
            blur_block=201,
            blur_width=50,
            block_size=501,
            relative_threshold=+5,
        ),
        BeamDetectionConfig(
            scaling_factor=8,
            roi_size=600,
            blur_block=301,
            blur_width=75,
            block_size=501,
            relative_threshold=+3.5,
        ),
    ),
}

def preprocess_images(load_type: Literal['lyse', 'h5'], h5_path: Optional[StrPath] = None, fig_debug: Optional[Figure] = None):
    preproc = BeamImagePreprocessor(
        la_align_detection_configs,
        load_type=load_type,
        h5_path=h5_path,
    )
    try:
        preproc.fit_beam_locations(fig_debug)
    except ValueError:
        print('oopsy no fit')
    return preproc


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    fig_debug = plt.figure(figsize=(10, 10), layout='constrained')
    # preproc = preprocess_images(load_type='lyse', fig_debug=fig_debug)
    preproc = BeamImagePreprocessor(
        la_align_detection_configs,
        load_type='lyse',
    )
    try:
        preproc.process_shot(fig_debug)
    except ValueError:
        print('oopsy no fit')
    fig = plt.figure(figsize=(6, 6), layout='constrained')
    preproc.plot_beam_locations(fig, crosshairs='all')
    preproc.h5_path.with_name('doneflag.txt').touch()
