import itertools
import logging
from collections.abc import Iterable, Mapping
from typing import ClassVar, Literal, NamedTuple, Optional, Union, cast

import numpy as np
import uncertainties
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from uncertainties import unumpy as unp

import h5py

from analysislib.common.analysis_config import (
    ImagingSystem,
    manta_local_addr_align_system,
    manta_tweezer_system,
)
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
    
    # fmt: off
    calibration_matrices: ClassVar[dict[float, NDArray]] = {
        10: np.array([
            [-5.45786373e+01, -5.80006349e+00, -3.02090656e+01, -3.14879954e+00],
            [-7.20055001e-01, -1.61494764e+01,  1.64580944e+00, -2.58167949e+01],
            [ 4.84636929e+00,  4.19192123e-01,  5.30617600e+00,  2.78971054e-02],
            [-2.01117198e-01,  5.13286733e-01, -3.31150327e-01,  3.17914301e+00],
        ]),
        2: np.array([
            [-8.76489713e+00, -9.82237294e-01, -5.31897975e+00, -5.74300889e-01],
            [ 4.17014015e-01, -2.74255103e+00,  3.51430352e-01, -3.95787614e+00],
            [ 6.65295442e-01,  7.94262309e-03,  8.39548104e-01,  1.20939191e-02],
            [ 2.15401425e-01,  1.90916659e-01,  1.25663462e-01,  5.37215736e-01],
        ])
    }
    """Calibration matrices; rows are the camera displacements
    [im_dx, im_dy, co_dx, co_dy] in px (image plane cam and collimated plane cam),
    while columns are [1h, 1v, 2h, 2v] in seconds.

    The camera displacements were formerly known as [tw_dx, tw_dy, la_dx, la_dy].
    """
    # fmt: on

    def __init__(
        self,
        detection_configs: Mapping[
            ImagingSystem, Union[BeamDetectionConfig, tuple[BeamDetectionConfig, ...]]
        ],
        load_type: Literal['lyse', 'h5'],
        h5_path: Optional[StrPath] = None,
    ):
        super().__init__(list(detection_configs), load_type, h5_path)
        self.detection_configs = detection_configs

    def _fit_beam_location(
        self, imaging_system: ImagingSystem, index: int, fig_debug: Optional[Figure]
    ):
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
            subfigs = fig_debug.subfigures(
                nrows=1,
                ncols=sum(len(exposure) for exposure in self.exposures_dict.values()),
            )
            subfigs_iter = iter(subfigs)
        else:
            subfigs_iter = itertools.repeat(None)
        self.fit_parameters = {
            imaging_system: tuple(
                self._fit_beam_location(imaging_system, i, next(subfigs_iter))
                for i in range(len(exposures))
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
            manta_local_addr_align_system: True,
        },
    ):
        axs = fig.subplots(
            nrows=max(len(exposures) for exposures in self.exposures_dict.values()),
            ncols=len(self.detection_configs),
            sharex='col',
            sharey='col',
            squeeze=False,
        )

        min_halfsizes = {
            manta_tweezer_system: 50,
            manta_local_addr_align_system: 200,
        }
        markers = ['+', 'x']
        for col, (imaging_system, exposures) in enumerate(self.exposures_dict.items()):
            axs[0, col].set_title(imaging_system.camera.image_group_name)

            for i, exposure in enumerate(exposures):
                ax = cast(Axes, axs[i, col])
                ax.imshow(exposure, cmap='Purples')

                if not self.fitted():
                    continue

                x_u, y_u, *_ = self.fit_parameters[imaging_system][i]

                crosshair_axs: Iterable[Axes] = []
                if crosshairs == 'all':
                    crosshair_axs = axs[:, col]
                elif crosshairs == 'single':
                    crosshair_axs = [ax]

                for crosshair_ax in crosshair_axs:
                    if crosshair_ax == ax:
                        label = f'({x_u:S}, {y_u:S}) px'
                    else:
                        label = None

                    crosshair_ax.plot(
                        [x_u.n],
                        [y_u.n],
                        marker=markers[i],
                        color='red',
                        label=label,
                        linestyle='None',
                    )

            if self.fitted() and crop[imaging_system]:
                padding_fraction = 1.15

                fit_params = self.fit_parameters[imaging_system]
                xy_vals = unp.nominal_values(
                    [single_fit_params[0:2] for single_fit_params in fit_params]
                )
                minvals = xy_vals.min(axis=0)
                maxvals = xy_vals.max(axis=0)
                xmean, ymean = (maxvals + minvals) / 2
                halfrange = max(
                    min_halfsizes[imaging_system],
                    padding_fraction * (maxvals - minvals).max() / 2,
                )

                # ylim order reversed for consistency with imshow default
                axs[0, col].set_xlim(xmean - halfrange, xmean + halfrange)
                axs[0, col].set_ylim(ymean + halfrange, ymean - halfrange)

        if self.parameters['do_local_addr_move']:
            axs[0, 0].set_ylabel('Before')
            axs[1, 0].set_ylabel('After')

            drive_voltage = self.parameters['local_addr_piezo_voltage']
            move_times = np.array([
                self.parameters['local_addr_piezo_dur_1h'],
                self.parameters['local_addr_piezo_dur_1v'],
                self.parameters['local_addr_piezo_dur_2h'],
                self.parameters['local_addr_piezo_dur_2v'],
            ])
            expected_shift = self.calibration_matrices[drive_voltage] @ move_times

            for i, imaging_setup in enumerate(self.imaging_setups):
                ax_col = axs[:, i]
                this_plane_expected_shift = expected_shift[2 * i : 2 * i + 2]
                initial_point = unp.nominal_values(self.fit_parameters[imaging_setup][0][0:2])
                expected_final_point = initial_point + this_plane_expected_shift
                print(initial_point)
                for ax in ax_col:
                    ax = cast(Axes, ax)

                    ax.annotate(
                        text='',
                        xy=expected_final_point,
                        xytext=initial_point,
                        arrowprops=dict(color='0.3', shrink=0, frac=0.05, headwidth=5, width=1, alpha=0.5)
                    )

        elif self.parameters['do_local_addr_alignment_check']:
            axs[0, 0].set_ylabel('Local addr')
            axs[1, 0].set_ylabel('Tweezers')

        if crosshairs:
            for ax in axs.flatten():
                ax.legend()

    # TODO consoliate this with BulkGasPreprocessor.process_shot()?
    def process_shot(self, fig_debug: Optional[Figure] = None):
        self.fit_beam_locations(fig_debug)
        gauss_nom = np.array([
            [
                unp.nominal_values(exposure_upopt)
                for exposure_upopt in fit_params
            ]
            for imaging_system, fit_params in self.fit_parameters.items()
        ])
        gauss_cov = np.array([
            [
                uncertainties.covariance_matrix(exposure_upopt)
                for exposure_upopt in fit_params
            ]
            for imaging_system, fit_params in self.fit_parameters.items()
        ])
        # assumes equal number of exposures per camera!
        n_exposures = len(next(iter(self.exposures_dict.values())))

        run_number = self.run_number
        fname = self.h5_path.with_name('beamspot_preprocess.h5')
        if run_number == 0:
            with h5py.File(fname, 'w') as f:
                f.attrs['n_runs'] = self.n_runs

                n_params = 6
                f.create_dataset(
                    'gaussian_spot_params_nom',
                    data=[gauss_nom],
                    maxshape=(
                        self.n_runs,
                        len(self.imaging_setups),
                        n_exposures,
                        n_params,
                    ),
                )
                f['gaussian_spot_params_nom'].attrs['fields'] = [
                    'x',
                    'y',
                    'width',
                    'height',
                    'amplitude',
                    'offset',
                ]
                f['gaussian_spot_params_nom'].attrs['units'] = [
                    'px',
                    'px',
                    'px',
                    'px',
                    'counts',
                    'counts',
                ]
                f['gaussian_spot_params_nom'].attrs['imaging_systems'] = [
                    system.camera.image_group_name for system in self.imaging_setups
                ]
                f.create_dataset(
                    'gaussian_spot_params_cov',
                    data=[gauss_cov],
                    maxshape=(
                        self.n_runs,
                        len(self.imaging_setups),
                        n_exposures,
                        n_params,
                        n_params,
                    ),
                )

            f.create_dataset(
                'current_params',
                data=self.current_params[np.newaxis, ...],
                maxshape=(self.n_runs, len(self.current_params)),
                chunks=True,
            )

            param_list = []
            for key in self.params.keys():
                param_list.append([key, self.params[key][1], self.params[key][0]])
            f.create_dataset('params', data=param_list)
        else:
            with h5py.File(fname, 'a') as f:
                f['gaussian_spot_params_nom'].resize(run_number + 1, axis=0)
                f['gaussian_spot_params_nom'][run_number] = gauss_nom
                f['gaussian_spot_params_cov'].resize(run_number + 1, axis=0)
                f['gaussian_spot_params_cov'][run_number] = gauss_cov
