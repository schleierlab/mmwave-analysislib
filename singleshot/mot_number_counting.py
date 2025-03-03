from analysis_lib import bulk_gas_analysis

roi_x = [550, 1350]#roi_x = [850, 1250] # Region of interest of X direction
roi_y = [650, 1450] #[750, 1150] # Region of interest of Y direction
roi_x_bkg = [1900, 2400] # Region of interest of X direction
roi_y_bkg= [1900, 2400] # Region of interest of Y direction
roi = [roi_x, roi_y, roi_x_bkg, roi_y_bkg]

bulk_gas_analysis_obj = bulk_gas_analysis()
i = 50 #mm lens before camera
o = 125 #mm lens focusing to atoms
d = 2.54*10 #mm, 1 inch lens diameter
exposure_time = 1e-3
bulk_gas_analysis_obj.get_magnification(i, o)
bulk_gas_analysis_obj.get_counts_per_atom(d, o, t = exposure_time)

bulk_gas_analysis_obj.load_image_in_lyse()
bulk_gas_analysis_obj.get_image_bkg_sub()
bulk_gas_analysis_obj.get_images_roi(roi)
bulk_gas_analysis_obj.get_atom_number()
bulk_gas_analysis_obj.plot_images()
bulk_gas_analysis_obj.save_atom_number()


