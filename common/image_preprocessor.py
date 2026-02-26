from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Literal

import numpy as np
import yaml

import h5py

from analysislib.analysis.data import h5lyze as hz
from analysislib.common.analysis_config import ImagingSystem
from analysislib.common.typing import StrPath


class ImagePreprocessor:
    run_number: int
    h5_path: Path
    exposures_dict: dict[ImagingSystem, tuple[np.ndarray, ...]]
    """All exposures taken during this shot. Indexed by the imaging system used."""

    parameters: dict[str, Any]
    """The effective labscript globals used, which includes default values if not supplied in runmanager"""

    n_runs: int
    '''Total number of runs for this runmanager expansion.'''

    run_time: str
    '''Time of the run, as a string of the form 20250722T153214 (strftime format %Y%m%dT%H%M%S)'''

    def __init__(
            self,
            imaging_setups: Sequence[ImagingSystem],
            load_type: Literal['lyse', 'h5'] = 'lyse',
            h5_path: Optional[StrPath] = None,
    ):
        """
        Base class for image preprocessing.
        This class handles the basic image loading and preprocessing operations
        common to both tweezer and bulk gas analysis.

        Parameters
        ----------
        imaging_setups : sequence of ImagingSystem
            Imaging setup configurations for all cameras used in the shot.
        load_type : str, default='lyse'
            'lyse' for h5 file active in lyse, 'h5' for h5 file with input h5_path
        h5_path : str, optional
            Path to h5 file, only used if load_type='h5'
        """
        self.imaging_setups = imaging_setups
        self.h5_path = self.get_h5_path(load_type=load_type, h5_path=h5_path)
        self.exposures_dict, self.run_number, self.globals, self.default_params = self.load_images()
        self.params, self.n_rep, self.current_params = self.get_scanning_params()

        self.parameters = self.default_params | self.globals

        with h5py.File(self.h5_path, mode='r') as f:
            self.n_runs = f.attrs['n_runs']
            self.run_time = f.attrs['run time']

    def get_h5_path(self, load_type: Literal['lyse', 'h5'], h5_path: Optional[StrPath] = None) -> Path:
        """
        get h5_path based on load_type
        Parameters
        ----------
        load_type: str
            'lyse' for h5 file active in lyse, 'h5' for h5 file with input h5_path
        h5_path: str
            path to h5 file, only used if load_type='h5'

        Returns
        -------
        h5_path: Path
            The actual h5 file path used based on the selected load_type
        """
        if load_type == 'lyse':
            import lyse
            # Is this script being run from within an interactive lyse session?
            if lyse.spinning_top:
                # If so, use the filepath of the current h5_path
                h5_path = Path(lyse.path)
            else:
                # If not, get the filepath of the last h5_path of the lyse DataFrame
                df = lyse.data()
                h5_path = Path(df.filepath.iloc[-1])
        elif load_type == 'h5':
            if h5_path is None:
                raise ValueError("When load_type is h5, please provide exact h5 path")
            h5_path = Path(h5_path)
        else:
            raise ValueError("load_type must be 'lyse' or 'h5'")

        return h5_path

    # TODO unit tests for this
    def get_scanning_params(self,):
        """
        get scanning parameters and number of repetitions based
        on the expansion of each globals, if the expansion is 'inner' or 'outer'
        then it is a scanning parameter, number of repetitions is determined by n_shots
        in globals. If only n_shots is present, then n_rep = 1 and scanning params = n_shots

        Returns
        -------
        params: dict
            Scanning parameters; key is the parameter name
            and value is a tuple (parameter_values, unit)
        n_rep: int
            number of repetitions
        """
        h5_path = self.h5_path
        repetition_param_name = 'repetition_index'
        with h5py.File(h5_path, mode='r+') as f:
            globals = f['globals']
            params = {}
            current_params = []
            for group in globals.keys():
                expansion = hz.getAttributeDict(globals[group]['expansion'])
                for key, value in expansion.items():
                    if value == 'outer' or value == 'inner':
                        global_var = hz.getAttributeDict(globals[group])[key]
                        global_unit = hz.getAttributeDict(globals[group]['units'])[key]
                        params[key] = (global_var, global_unit)
            if repetition_param_name in params:
                rep_str, _ = params[repetition_param_name]
                if rep_str[0:2] != 'np':
                    rep_str = 'np.' + rep_str
                rep = eval(rep_str)
                n_rep = rep.shape[0]
                del params[repetition_param_name]
                if len([key for key in params if key[:3] != 'do_']) == 0:
                    params[repetition_param_name] = [rep_str, 'Shots']
                    for key in [key for key in params if key[:3] == 'do_']:
                        del params[key]
                    n_rep = 1
            else:
                n_rep = 1
            for key, value in params.items():
                current_params.append(self.globals[key])

        return params, n_rep, np.array(current_params)

    def get_current_scanning_params(self,):
        """
        get current scanning parameters from globals

        Returns
        -------
        current_param_vals: dict
            Current scanning parameters; key is the parameter name
            and value is the parameter value
        """

    @staticmethod
    def get_scanning_params_static(h5_path):
        """
        get scanning parameters from globals

        Returns
        -------
        params: list
        """
        with h5py.File(h5_path, mode='r+') as f:
            globals = f['globals']
            params = {}
            for group in globals.keys():
                expansion = hz.getAttributeDict(globals[group]['expansion'])
                for key, value in expansion.items():
                    if value == 'outer' or value == 'inner':
                        global_var = hz.getAttributeDict(globals[group])[key]
                        global_unit = hz.getAttributeDict(globals[group]['units'])[key]
                        params[key] = [global_var, global_unit]

            rep_str =params['n_shots'][0]
            if rep_str[0:2] != 'np':
                rep_str = 'np.' + rep_str
            rep = eval(rep_str)
            n_rep = rep.shape[0]
            del params['n_shots']
            if len(params) == 0:
                params['n_shots'] = [rep_str,'Shots']
                n_rep = 1

        return params, n_rep

    def load_images(self) -> tuple[dict[ImagingSystem, tuple[np.ndarray, ...]], int, dict[str, Any], dict[str, Any]]:
        """
        load image inside the h5 file, return current run number and globals

        Returns
        -------
        images: dictionary with keys based on different imaging cameras used
            images.keys(): list of all keys of images, which is based on different imaging cameras
            images.values(): list of all values of images, array like, shape [n_px, n_px]
        run_number: int
            current run number
        globals, default_params: dict
            Dicts for runmanager-set and default parameter values.
        """
        with h5py.File(self.h5_path, mode='r+') as f:
            globals_dict = hz.getGlobalsFromFile(self.h5_path)
            hz.getDefaultParamsFromFile

            images_dict: dict[ImagingSystem, tuple[np.ndarray, ...]] = dict()
            for imaging_setup in self.imaging_setups:
                single_camera_images = hz.datasetsToDictionary(f[imaging_setup.camera.image_group_name], recursive=True)

                images_dict[imaging_setup] = tuple(
                    single_camera_images[imaging_setup.camera.image_name_stem + str(i)]
                    for i in range(len(single_camera_images))
                )

            run_number = f.attrs['run number']
            try:
                default_params = hz.getDefaultParamsFromFile(self.h5_path)
            except KeyError:
                default_params = dict()

        return images_dict, run_number, globals_dict, default_params
    
    @staticmethod
    def _load_default_params_from_yaml(defaul_params_path: Path):
        """
        Load default parameters from YAML file.

        Parameters
        ----------
        defaul_params_path : Path
            Path to the YAML file containing the default parameters.

        Returns
        -------
        default_params : dict
            Dictionary of default parameters.
        """
        with defaul_params_path.open('rt') as stream:
            default_params = yaml.safe_load(stream)

        return default_params

    def _load_ylims_from_globals(self):
        """Load the atom ROI y-coordinates from globals. Sometimes they change...
        # TODO: further explanation

        Returns
        -------
        atom_roi_ylims : list
            List of y-coordinates for the atom ROI.
        """
        try:
            atom_roi_ymin, atom_roi_height = self.globals["kinetix_roi_row"]
            atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
        except KeyError:
            try:
                default_params = self._load_default_params_from_yaml(self.DEFAULT_PARAMS_PATH)
                atom_roi_ymin, atom_roi_height  = np.array(eval(default_params["Tweezers"]["kinetix_roi_row"]['value']))
                atom_roi_ylims = [atom_roi_ymin, atom_roi_ymin + atom_roi_height]
            except KeyError:
                raise KeyError('kinetix_roi_row not found in globals')
        return atom_roi_ylims
