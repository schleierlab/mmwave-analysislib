from common.analysis_config import manta_system, BulkGasAnalysisConfig
from common.bulk_gas_analysis import BulkGasPreprocessor
from common.bulk_gas_plot import BulkGasPlotter
from common.image import ROI


config = BulkGasAnalysisConfig(
    imaging_system=manta_system,
    exposure_time=1e-3,
    atoms_roi=ROI(xmin=800, xmax=1100, ymin=900, ymax=1200),
    bkg_roi=ROI(xmin=1900, xmax=2048, ymin=1900, ymax=2048),
)

bulk_gas_analysis_obj = BulkGasPreprocessor(config)
preprocess_file = bulk_gas_analysis_obj.process_shot()

# bulk_gas_analysis_obj.get_atom_number()
# bulk_gas_analysis_obj.plot_images()
# bulk_gas_analysis_obj.plot_atom_number()

plotter = BulkGasPlotter(preprocess_file, bulk_gas_analysis_obj)
plotter.plot_atom_number()
plotter.plot_images()
