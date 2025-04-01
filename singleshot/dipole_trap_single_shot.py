import matplotlib.pyplot as plt

from analysislib.common.analysis_config import kinetix_system, BulkGasAnalysisConfig
from analysislib.common.bulk_gas_preproc import BulkGasPreprocessor
from analysislib.common.bulk_gas_statistics import BulkGasStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig


analysis_config = BulkGasAnalysisConfig(
    imaging_system=kinetix_system,
    exposure_time=80e-3,
    atoms_roi=ROI(xmin=1100, xmax=1550, ymin=1125, ymax=1275),#atoms_roi=ROI(xmin=1000, xmax=1950, ymin=1125, ymax=1275),
    bkg_roi=ROI(xmin=1900, xmax=2400, ymin=1900, ymax=2400),
)

bulk_gas_preproc = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
)

processed_results_fname = bulk_gas_preproc.process_shot()

fig = plt.figure(layout = "constrained", figsize = [10,4])
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

bulk_gas_preproc.show_images(fig = subfigs[0], raw_img_scale = 800)

plotter = BulkGasStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=bulk_gas_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)

plotter.plot_atom_number(fig = subfigs[1], plot_lorentz = False)
