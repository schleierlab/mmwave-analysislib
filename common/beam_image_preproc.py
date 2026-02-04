import itertools
import logging
from collections.abc import Iterable, Mapping
from typing import Literal, NamedTuple, Optional, Union, cast

import numpy as np
import uncertainties
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from uncertainties import unumpy as unp

from analysislib.common.analysis_config import ImagingSystem, manta_local_addr_align_system, manta_tweezer_system
from analysislib.common.image import Image
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.typing import StrPath


logger = logging.getLogger(__name__)


class BeamDetectionConfig(NamedTuple):
    scaling_factor: float
    roi_size: int
    blur_block: int
    blur_width: float
    block_size: int
    relative_threshold: float


class BeamImagePreprocessor(ImagePreprocessor):
    fit_parameters: dict[ImagingSystem, tuple[NDArray, ...]]

    def __init__(
        self,
        detection_configs: Mapping[ImagingSystem, Union[BeamDetectionConfig, tuple[BeamDetectionConfig, ...]]],
        load_type: Literal['lyse', 'h5'],
        h5_path: Optional[StrPath] = None,
    ):
        super().__init__(list(detection_configs), load_type, h5_path)
        self.detection_configs = detection_configs

    def _fit_beam_location(self, imaging_system: ImagingSystem, index: int, fig_debug: Optional[Figure]):
        detection_config_spec = self.detection_configs[imaging_system]
        if isinstance(detection_config_spec, BeamDetectionConfig):
            detection_config = detection_config_spec
        else:
            detection_config = detection_config_spec[index]

        img_arr = self.exposures_dict[imaging_system][index]

        if img_arr.max() / detection_config.scaling_factor > 255:
            logging.warning(
                f'Out of bounds, clipping! {self.h5_path=}, {imaging_system.camera.image_group_name=}, {index=}, {img_arr.max()=}'
            )
        img = Image(
            np.clip(img_arr / detection_config.scaling_factor, 0, 255).astype(np.uint8)
        )

        finder = TweezerFinder(img)
        try:
            (roi,) = finder.detect_rois_by_contours(
                roi_number=1,
                roi_size=detection_config.roi_size,
                blur_block=detection_config.blur_block,
                blur_width=detection_config.blur_width,
                block_size=detection_config.block_size,
                relative_threshold=detection_config.relative_threshold,
            )
        except ValueError as e:
            logger.exception(e)
            raise e
        finally:
            if fig_debug is not None:
                finder.plot_contour_site_detection(fig_debug)
        
        popt, pcov = img.roi_fit_gaussian2d(roi)
        upopt = uncertainties.correlated_values(popt, pcov)

        return upopt

    def fit_beam_locations(self, fig_debug: Optional[Figure] = None):
        if fig_debug is not None:
            subfigs = fig_debug.subfigures(nrows=1, ncols=sum(len(exposure) for exposure in self.exposures_dict.values()))
            subfigs_iter = iter(subfigs)
        else:
            subfigs_iter = itertools.repeat(None)
        self.fit_parameters = {
            imaging_system: tuple(
                self._fit_beam_location(imaging_system, i, next(subfigs_iter)) for i in range(len(exposures))
            )
            for imaging_system, exposures in self.exposures_dict.items()
        }

    def fitted(self) -> bool:
        return hasattr(self, 'fit_parameters')

    def plot_beam_locations(
            self,
            fig: Figure,
            crosshairs: Literal[False, 'single', 'all'] = False,
            crop: Mapping[ImagingSystem, bool] = {
                manta_tweezer_system: True,
                manta_local_addr_align_system: False,
            },
    ):
        axs = fig.subplots(
            nrows=max(len(exposures) for exposures in self.exposures_dict.values()),
            ncols=len(self.detection_configs),
            sharex='col',
            sharey='col',
            squeeze=False,
        )

        markers = ['+', 'x']
        for col, (imaging_system, exposures) in enumerate(self.exposures_dict.items()):
            axs[0, col].set_title(imaging_system.camera.image_group_name)

            for i, exposure in enumerate(exposures):
                ax = cast(Axes, axs[i, col])
                ax.imshow(exposure, cmap='Purples')

                if not self.fitted():
                    continue

                x_u, y_u, *_ = self.fit_parameters[imaging_system][i]
                ax.set_title(f'Fit location: ({x_u:S}, {y_u:S}) px')

                crosshair_axs: Iterable[Axes] = []
                if crosshairs == 'all':
                    crosshair_axs = axs[:, col]
                elif crosshairs == 'single':
                    crosshair_axs = [ax]

                for crosshair_ax in crosshair_axs:
                    crosshair_ax.plot(
                        [x_u.n],
                        [y_u.n],
                        marker=markers[i],
                        color='red',
                    )

            if self.fitted() and crop[imaging_system]:
                padding_fraction = 1.15

                fit_params = self.fit_parameters[imaging_system]
                xy_vals = unp.nominal_values([single_fit_params[0:2] for single_fit_params in fit_params])
                minvals = xy_vals.min(axis=0)
                maxvals = xy_vals.max(axis=0)
                xmean, ymean = (maxvals + minvals) / 2
                halfrange = max(100, padding_fraction * (maxvals - minvals).max() / 2)

                # ylim order reversed for consistency with imshow default
                axs[0, col].set_xlim(xmean - halfrange, xmean + halfrange)
                axs[0, col].set_ylim(ymean + halfrange, ymean - halfrange)
