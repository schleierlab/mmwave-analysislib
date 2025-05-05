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
    atoms_roi=ROI(xmin=800, xmax=1100, ymin=900, ymax=1200),
    bkg_roi=ROI(xmin=1900, xmax=2048, ymin=1900, ymax=2048),
)

bulk_gas_preproc = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
)

if DO_GAUSSIAN_FIT:
    processed_results_fname = bulk_gas_preproc.process_shot(cloud_fit='gaussian_uniform')
    # cloud_fit='gaussian' or 'gaussian_uniform'
else:
    processed_results_fname = bulk_gas_preproc.process_shot()


fig = plt.figure(layout = "constrained", figsize = [10, 4])
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

bulk_gas_preproc.show_images(fig = subfigs[0])

plotter = BulkGasStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=bulk_gas_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)
if DO_GAUSSIAN_FIT:
    plotter.plot_mot_params(fig = subfigs[1], uniform=True)
else:
    plotter.plot_atom_number(fig = subfigs[1], plot_lorentz=False)

fig.savefig(bulk_gas_preproc.h5_path.with_name("mot_single_shot.pdf"))
