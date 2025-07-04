import matplotlib.pyplot as plt

from analysislib.common.analysis_config import kinetix_system, BulkGasAnalysisConfig
from analysislib.common.bulk_gas_preproc import BulkGasPreprocessor
from analysislib.common.bulk_gas_statistics import BulkGasStatistician
from analysislib.common.image import ROI
from analysislib.common.plot_config import PlotConfig


analysis_config = BulkGasAnalysisConfig(
    imaging_system=kinetix_system,
    exposure_time=80e-3,
    atoms_roi=ROI(xmin=800, xmax=1650, ymin=1000, ymax=1400),  # (xmin=800, xmax=1650, ymin=1000, ymax=1400)
    bkg_roi=ROI(xmin=1900, xmax=2400, ymin=1900, ymax=2400),
)

bulk_gas_preproc = BulkGasPreprocessor(
    config=analysis_config,
    load_type='lyse',
    h5_path=None,
)

processed_results_fname = bulk_gas_preproc.process_shot()

fig = plt.figure(layout = "constrained", figsize = [10, 4])
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07)

bulk_gas_preproc.show_images(fig = subfigs[0], raw_img_scale = 800)

fig2 = plt.figure(layout='constrained', figsize=[6, 6])
# bulk_gas_preproc.show_state_sensitive_images(fig2)

plotter = BulkGasStatistician(
    preproc_h5_path=processed_results_fname,
    shot_h5_path=bulk_gas_preproc.h5_path, # Used only for MLOOP
    plot_config=PlotConfig(),
)

plotter.plot_atom_number(fig=subfigs[1], plot_lorentz=False)

fig.savefig(bulk_gas_preproc.h5_path.with_name("dipole_trap_single_shot.pdf"))
