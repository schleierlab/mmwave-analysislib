import logging
from pathlib import Path
from typing import cast

import h5py  # type: ignore
import numpy as np
import pandas as pd
import uncertainties

from analysislib.common.base_statistics import BaseStatistician
from analysislib.common.scanning_params import ScanningParameters
from analysislib.common.typing import StrPath


logger = logging.getLogger(__name__)


class BeamspotStatistician(BaseStatistician):
    def __init__(
            self,
            preproc_h5_path: StrPath,
            *,
            shot_index: int = -1,
    ):
        super().__init__(preproc_h5_path=preproc_h5_path, shot_index=shot_index)
        self._load_processed_quantities(preproc_h5_path)
        self.folder_path = Path(preproc_h5_path).parent

    def _load_processed_quantities(self, preproc_h5_path):
        """Load processed quantities from an h5 file.

        Parameters
        ----------
        preproc_h5_path : str
            Path to the processed quantities h5 file
        """
        with h5py.File(preproc_h5_path, 'r') as f:
            self.fit_param_names = list(f['gaussian_spot_params_nom'].attrs['fields'])
            self.gaussian_spot_params_nom = f['gaussian_spot_params_nom'][:]
            self.gaussian_spot_params_cov = f['gaussian_spot_params_cov'][:]
            # n_shots x n_img_systems x n_exposures x 6 (x 6 add'ly for covariance matrices)

            n_gaussian_params = self.gaussian_spot_params_nom.shape[-1]
            if self.gaussian_spot_params_cov.shape[-2:] != (n_gaussian_params, n_gaussian_params):
                raise ValueError

            self.params_list = f['params'][:]
            self.n_runs = cast(int, f.attrs['n_runs'])

            self.current_params = f['current_params'][:]
            # self.run_times_strs = np.char.decode(np.asarray(f['run_times'][:], dtype=bytes), encoding='utf-8')

            self.params = ScanningParameters.from_h5_tuples(self.params_list)

    def dataframe_u(self):
        noms = self.gaussian_spot_params_nom.reshape(-1, 6)
        covs = self.gaussian_spot_params_cov.reshape(-1, 6, 6)
        arr = np.array([
            uncertainties.correlated_values(n, c)
            for n, c in zip(noms, covs)
        ]).reshape(self.gaussian_spot_params_nom.shape)

        nshots, ncams, nexpos, nparams = arr.shape
        mi = pd.MultiIndex.from_product(
            [range(nshots), ('fo', 'co'), range(nexpos)],
            names=('shot', 'camera', 'exposure'),
        )
        return pd.DataFrame.from_records(
            arr.reshape(-1, 6),
            index=mi,
            columns=self.fit_param_names,
        )

    def calibrated_move_matrix(self):
        raise NotImplementedError

    @property
    def shots_processed(self) -> int:
        return self.gaussian_spot_params_nom.shape[0]
