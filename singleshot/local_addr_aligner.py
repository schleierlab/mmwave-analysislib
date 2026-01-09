import matplotlib.pyplot as plt

from analysislib.common.analysis_config import manta_local_addr_align_system, BulkGasAnalysisConfig
from analysislib.common.bulk_gas_preproc import BulkGasPreprocessor
from analysislib.common.bulk_gas_statistics import BulkGasStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig

DO_GAUSSIAN_FIT = True
analysis_config = BulkGasAnalysisConfig(
    imaging_system=manta_local_addr_align_system,
    exposure_time=1e-3,
    atoms_roi=ROI(xmin=0, xmax=2048, ymin=0, ymax=2048),#(xmin=300, xmax=1500, ymin=450, ymax=1050),
    bkg_roi=ROI(xmin=1900, xmax=2048, ymin=1900, ymax=2048),
)

bulk_gas_preproc = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
    background = False,
    beam_image=True
)

processed_results_fname = bulk_gas_preproc.process_shot(cloud_fit='gaussian')

fig = plt.figure(layout = "constrained", figsize = [10, 4])
subfigs = fig.subfigures(nrows=2, ncols=1, wspace=0.07)

bulk_gas_preproc.show_all_images(fig = subfigs[0])

plotter = BulkGasStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=bulk_gas_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
    multi_image=True
)
plotter.plot_beam_positions(fig = subfigs[1])
