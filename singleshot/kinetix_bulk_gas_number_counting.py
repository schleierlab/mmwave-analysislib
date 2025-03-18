from analysis_lib import BulkGasAnalysis, kinetix_setup
import matplotlib.pyplot as plt

roi_x =  [1100, 1700]# Region of interest of X direction
roi_y = [1160, 1220] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of background of X direction
roi_y_bkg= [1900, 2400] # Region of background of Y direction

t_expo = 80e-3 #s, exposure time

bulk_gas_analysis_obj = BulkGasAnalysis(
    imaging_setup=kinetix_setup, # camera is set to kinetix
    exposure_time=t_expo,
    atoms_roi=[roi_x, roi_y],
    bkg_roi=[roi_x_bkg, roi_y_bkg],
)

bulk_gas_analysis_obj.get_atom_number()
bulk_gas_analysis_obj.plot_images(raw_image_scale=200, roi_image_scale=100)
bulk_gas_analysis_obj.plot_atom_number()
bulk_gas_analysis_obj.plot_amplitude_vs_parameter()