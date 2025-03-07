from analysis_lib import TweezerAnalysis, kinetix_path
import matplotlib.pyplot as plt

roi_x = [1200, 1450]#roi_x = [850, 1250] # Region of interest of X direction
roi_y = [1080, 1100] #[750, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction

t_expo = 80e-3 #s, exposre time

bulk_gas_analysis_obj = TweezerAnalysis(
    imaging_setup=kinetix_path,
    exposure_time=t_expo,
    atoms_roi=[roi_x, roi_y],
    bkg_roi=[roi_x_bkg, roi_y_bkg],
)

bulk_gas_analysis_obj.get_atom_number()
bulk_gas_analysis_obj.plot_images()
bulk_gas_analysis_obj.plot_atom_number()