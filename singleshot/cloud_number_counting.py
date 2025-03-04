from analysis_lib import BulkGasAnalysis, manta_path

roi_x = [550, 1350]#roi_x = [850, 1250] # Region of interest of X direction
roi_y = [650, 1450] #[750, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction

t_expo = 1e-3 #s, exposre time

bulk_gas_analysis_obj = BulkGasAnalysis(
    imaging_setup=manta_path,
    exposure_time=t_expo,
    atoms_roi=[roi_x, roi_y],
    bkg_roi=[roi_x_bkg, roi_y_bkg],
)

bulk_gas_analysis_obj.get_atom_number()
bulk_gas_analysis_obj.plot_images()
bulk_gas_analysis_obj.plot_atom_number()
