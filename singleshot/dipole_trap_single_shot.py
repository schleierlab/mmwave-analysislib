from common.analysis_config import kinetix_system, BulkGasAnalysisConfig
from common.bulk_gas_analysis import BulkGasPreprocessor
from common.bulk_gas_plot import BulkGasPlotter
from common.image import ROI
from common.plot_config import PlotConfig
from matplotlib import colors, patches, pyplot as plt


analysis_config = BulkGasAnalysisConfig(
    imaging_system=kinetix_system,
    exposure_time=80e-3,
    atoms_roi=ROI(xmin=1000, xmax=1800, ymin=1160, ymax=1350),
    bkg_roi=ROI(xmin=1900, xmax=2400, ymin=1900, ymax=2400),
)

bulk_gas_analyzer = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
)

processed_results_fname = bulk_gas_analyzer.process_shot()

fig = plt.figure(layout = "constrained", figsize = [10,4])
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

bulk_gas_analyzer.show_images(fig = subfigs[0], raw_img_scale = 800)

plotter = BulkGasPlotter(
    processed_results_fname, 
    plot_config=PlotConfig(),
)

plotter.plot_atom_number(fig = subfigs[1])
