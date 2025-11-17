import matplotlib.pyplot as plt

from analysislib.common.analysis_config import manta_system, BulkGasAnalysisConfig
from analysislib.common.bulk_gas_preproc import BulkGasPreprocessor
from analysislib.common.bulk_gas_statistics import BulkGasStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig

DO_GAUSSIAN_FIT = True
analysis_config = BulkGasAnalysisConfig(
    imaging_system=manta_system,
    exposure_time=1e-3,
    atoms_roi=ROI(xmin=0, xmax=2048, ymin=0, ymax=2048),
    bkg_roi=ROI(xmin=1900, xmax=2048, ymin=1900, ymax=2048),
)

bulk_gas_preproc = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
    background = False
)

processed_results_fname = bulk_gas_preproc.process_shot(cloud_fit='gaussian_uniform')

bulk_gas_preproc.show_images(fig = subfigs[0])