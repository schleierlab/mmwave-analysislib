from analysis_lib import BulkGasAnalysis, manta_path

roi_x = [900, 1150] # Region of interest of X direction
roi_y = [900, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of background of X direction
roi_y_bkg= [1900, 2400] # Region of background of Y direction

t_expo = 1e-3 #s, exposre time

gaussian_fit_params = [
                25.65785665, # amplitude
                167.40629676, # x center
                78.80645476, # y center
                39.60946011, # x waist
                32.51922039, # y waist
                -0.51510428, # rotation
                1.7804111, # offset
            ]
# This is the gaussian fit parameters, the center, waist, rotation and offset is fixed,
# The only parameter that is free is the amplitude

bulk_gas_analysis_obj = BulkGasAnalysis(
    imaging_setup=manta_path,
    exposure_time=t_expo,
    atoms_roi=[roi_x, roi_y],
    bkg_roi=[roi_x_bkg, roi_y_bkg],
)


bulk_gas_analysis_obj.get_atom_gaussian_fit(option="amplitude only", gaussian_fit_params=gaussian_fit_params)
bulk_gas_analysis_obj.plot_images()
bulk_gas_analysis_obj.plot_amplitude_vs_parameter()
