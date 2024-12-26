"""
Utility functions for loading and analyzing Kinetix camera images from H5 files.
These functions are designed to be generic and reusable across different analysis scripts.

Created on: 2024-12-26
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from pathlib import Path
from analysis.data import h5lyze as hz

def get_h5_path(lyse):
    """
    Get the H5 file path based on lyse session.
    
    Args:
        lyse: lyse module instance
        
    Returns:
        str: Path to the H5 file
    """
    if lyse.spinning_top:
        return lyse.path
    else:
        df = lyse.data()
        return df.filepath.iloc[-1]

def load_h5_data(h5_path):
    """
    Load data from H5 file.
    
    Args:
        h5_path (str): Path to the H5 file
        
    Returns:
        tuple: (images, kinetix_roi_row, loop_var, info_dict)
            - images: Dictionary containing image data
            - kinetix_roi_row: ROI row data from Kinetix camera
            - loop_var: Loop variable value
            - info_dict: Dictionary containing file information
    """
    with h5py.File(h5_path, mode='r+') as f:
        globals_dict = hz.attributesToDictionary(f['globals'])
        info_dict = hz.getAttributeDict(f)
        images = hz.datasetsToDictionary(f['kinetix_images'], recursive=True)
        kinetix_roi_row = np.array(f['globals'].attrs.get('kinetix_roi_row'))
        
        # Find looping global variable
        loop_glob = next((glob for group in globals_dict 
                         for glob in globals_dict[group] 
                         if globals_dict[group][glob][0:2] == "np"), None)
        
        try:
            loop_var = float(f['globals'].attrs.get(loop_glob))
        except:
            loop_var = info_dict.get('run number')
            
        return images, kinetix_roi_row, loop_var, info_dict

def load_roi_data(roi_paths, load_roi=True):
    """
    Load ROI data from files.
    
    Args:
        roi_paths (dict): Dictionary containing paths to ROI files
        load_roi (bool): Flag to determine if ROI should be loaded
        
    Returns:
        tuple: (site_roi_x, site_roi_y, roi_x) or (None, None, None) if load_roi is False
    """
    if not load_roi:
        return None, None, None
        
    site_roi_x = np.load(roi_paths['site_roi_x'])
    site_roi_y = np.load(roi_paths['site_roi_y'])
    roi_x = np.load(roi_paths['roi_x'])
    
    # Add minimum value as first entry and adjust y-offset
    site_roi_x = np.concatenate([[np.min(site_roi_x, axis=0)], site_roi_x])
    site_roi_y = np.concatenate([[np.min(site_roi_y, axis=0) + 10], site_roi_y])
    
    return site_roi_x, site_roi_y, roi_x

def analyze_site_signals(tweezer_roi, site_roi_x, site_roi_y, threshold):
    """
    Analyze signals at each tweezer site.
    
    Args:
        tweezer_roi (ndarray): Region of interest for tweezer array
        site_roi_x (ndarray): X coordinates of site ROIs
        site_roi_y (ndarray): Y coordinates of site ROIs
        threshold (float): Signal threshold for atom detection
        
    Returns:
        tuple: (rect_sig, atom_exist_lst, roi_number_lst)
            - rect_sig: List of rectangle patches for visualization
            - atom_exist_lst: List indicating atom presence (1) or absence (0)
            - roi_number_lst: List of signal sums for each ROI
    """
    rect_sig = []
    atom_exist_lst = []
    roi_number_lst = []
    
    for i in np.arange(site_roi_x.shape[0]):
        y_start, y_end = site_roi_y[i,0], site_roi_y[i,1]
        x_start, x_end = site_roi_x[i,0], site_roi_x[i,1]
        
        site_roi_signal = tweezer_roi[y_start:y_end, x_start:x_end]
        signal_sum = np.sum(site_roi_signal)
        roi_number_lst.append(signal_sum)
        
        if signal_sum > threshold:
            rect = patches.Rectangle(
                (x_start, y_start), 
                x_end - x_start, 
                y_end - y_start, 
                linewidth=1,
                edgecolor='gold',
                facecolor='none',
                alpha=0.5,
                fill=False)
            rect_sig.append(rect)
            atom_exist_lst.append(1)
        else:
            atom_exist_lst.append(0)
            
    return rect_sig, atom_exist_lst, roi_number_lst

def plot_results(tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2, 
                rect, rect_sig_1, rect_sig_2, show_site_roi=True):
    """
    Plot the analysis results.
    
    Args:
        tweezer_roi_1 (ndarray): First tweezer ROI
        bkg_roi_1 (ndarray): First background ROI
        tweezer_roi_2 (ndarray): Second tweezer ROI
        bkg_roi_2 (ndarray): Second background ROI
        rect (list): List of rectangle patches for all sites
        rect_sig_1 (list): List of rectangle patches for detected atoms in first image
        rect_sig_2 (list): List of rectangle patches for detected atoms in second image
        show_site_roi (bool): Flag to show site ROIs in plot
    """
    # Create figure with larger size
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 20), constrained_layout=True)
    ax_tweezer_1, ax_tweezer_2, ax_bkg_1, ax_bkg_2 = axs

    # Set larger font sizes
    plt.rcParams.update({'font.size': 14})
    fig.suptitle('Tweezer Array Imaging Analysis', fontsize=16)

    for ax in axs:
        ax.set_xlabel('x [px]', fontsize=14)
        ax.set_ylabel('y [px]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    roi_image_scale = 150
    roi_img_color_kw = dict(cmap='viridis', vmin=0, vmax=roi_image_scale)

    # Plot first tweezer ROI
    ax_tweezer_1.set_title('1st Tweezer ROI', fontsize=14, pad=10)
    pos = ax_tweezer_1.imshow(tweezer_roi_1, **roi_img_color_kw)
    if show_site_roi:
        pc = PatchCollection(rect, match_original=True, alpha=0.3)
        ax_tweezer_1.add_collection(pc)
        if rect_sig_1:
            pc_sig = PatchCollection(rect_sig_1, match_original=True)
            ax_tweezer_1.add_collection(pc_sig)
    cbar = fig.colorbar(pos, ax=ax_tweezer_1)
    cbar.ax.tick_params(labelsize=12)

    # Plot second tweezer ROI
    ax_tweezer_2.set_title('2nd Tweezer ROI', fontsize=14, pad=10)
    pos = ax_tweezer_2.imshow(tweezer_roi_2, **roi_img_color_kw)
    if show_site_roi:
        pc = PatchCollection(rect, match_original=True, alpha=0.3)
        ax_tweezer_2.add_collection(pc)
        if rect_sig_2:
            pc_sig = PatchCollection(rect_sig_2, match_original=True)
            ax_tweezer_2.add_collection(pc_sig)
    cbar = fig.colorbar(pos, ax=ax_tweezer_2)
    cbar.ax.tick_params(labelsize=12)

    # Plot first background ROI
    ax_bkg_1.set_title('1st Background ROI', fontsize=14, pad=10)
    pos = ax_bkg_1.imshow(bkg_roi_1, **roi_img_color_kw)
    cbar = fig.colorbar(pos, ax=ax_bkg_1)
    cbar.ax.tick_params(labelsize=12)

    # Plot second background ROI
    ax_bkg_2.set_title('2nd Background ROI', fontsize=14, pad=10)
    pos = ax_bkg_2.imshow(bkg_roi_2, **roi_img_color_kw)
    cbar = fig.colorbar(pos, ax=ax_bkg_2)
    cbar.ax.tick_params(labelsize=12)

def process_images(first_image, second_image, first_image_bkg, second_image_bkg, roi_config):
    """
    Process image pairs with background subtraction.
    
    Args:
        first_image (ndarray): First image
        second_image (ndarray): Second image
        first_image_bkg (ndarray): First background image
        second_image_bkg (ndarray): Second background image
        roi_config (dict): Dictionary containing ROI configuration
        
    Returns:
        tuple: (tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2)
    """
    try:
        sub_image1 = first_image - first_image_bkg
        sub_image2 = second_image - second_image_bkg
    except:
        raise ValueError('Error in background subtraction!')

    # Extract regions of interest for signal and background
    tweezer_roi_1 = sub_image1[:, roi_config['x'][0]:roi_config['x'][1]]
    bkg_roi_1 = sub_image1[:, roi_config['background_x'][0]:roi_config['background_x'][1]]

    tweezer_roi_2 = sub_image2[:, roi_config['x'][0]:roi_config['x'][1]]
    bkg_roi_2 = sub_image2[:, roi_config['background_x'][0]:roi_config['background_x'][1]]

    return tweezer_roi_1, bkg_roi_1, tweezer_roi_2, bkg_roi_2

def load_threshold(folder_path, load_threshold=True, default_threshold=746.8):
    """
    Load threshold value from file or use default.
    
    Args:
        folder_path (str): Path to the folder containing threshold file
        load_threshold (bool): Flag to determine if threshold should be loaded
        default_threshold (float): Default threshold value if not loading from file
        
    Returns:
        float: Threshold value for atom detection
    """
    if load_threshold:
        threshold = np.load(os.path.join(folder_path, "th.npy"))[0]
        print(f'threshold = {threshold} count')
    else:
        threshold = default_threshold
    return threshold

def load_average_background(h5_path):
    """
    Load pre-calculated average background images.
    
    Args:
        h5_path (str): Path to the H5 file
        
    Returns:
        tuple: (first_image_bkg, second_image_bkg)
            - first_image_bkg: Average background for first image
            - second_image_bkg: Average background for second image
    """
    avg_shot_bkg_file_path = Path(Path(h5_path).parent.parent, 'avg_shot_bkg.npy')
    try:
        avg_shot_bkg = np.load(avg_shot_bkg_file_path)
        return avg_shot_bkg[0,:,:], avg_shot_bkg[1,:,:]
    except FileNotFoundError:
        raise FileNotFoundError('Make sure you already have the averaged background!')

def create_site_rectangles(site_roi_x, site_roi_y):
    """
    Create rectangle patches for visualization.
    
    Args:
        site_roi_x (ndarray): X coordinates of site ROIs
        site_roi_y (ndarray): Y coordinates of site ROIs
        
    Returns:
        list: List of rectangle patches for visualization
    """
    rect = []
    for i in np.arange(site_roi_x.shape[0]):
        rect.append(patches.Rectangle(
            (site_roi_x[i,0], site_roi_y[i,0]),
            site_roi_x[i,1]-site_roi_x[i,0],
            site_roi_y[i,1]-site_roi_y[i,0],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
            alpha=0.3,
            fill=False))
    return rect

def calculate_survival_rate(atom_exist_lst_1, atom_exist_lst_2):
    """
    Calculate survival rate from atom existence lists.
    
    Args:
        atom_exist_lst_1 (ndarray): Array indicating atom presence in first image
        atom_exist_lst_2 (ndarray): Array indicating atom presence in second image
        
    Returns:
        float: Survival rate
    """
    return sum(1 for x,y in zip(atom_exist_lst_1, atom_exist_lst_2) 
              if x == 1 and y == 1) / np.sum(atom_exist_lst_1)

def prepare_roi_data(roi_number_lst_1, roi_number_lst_2):
    """
    Prepare ROI data for saving.
    
    Args:
        roi_number_lst_1 (list): List of ROI numbers from first image
        roi_number_lst_2 (list): List of ROI numbers from second image
        
    Returns:
        ndarray: Prepared ROI data array
    """
    roi_number_lst = np.row_stack([np.array(roi_number_lst_1), np.array(roi_number_lst_2)])
    return roi_number_lst.reshape((roi_number_lst.shape[0], roi_number_lst.shape[1], 1))

def save_data(folder_path, survival_rate, loop_var, roi_number_lst, run_number):
    """
    Save analysis results to files.
    
    Args:
        folder_path (str): Path to save the data files
        survival_rate (float): Calculated survival rate
        loop_var (float): Loop variable value
        roi_number_lst (ndarray): Array of ROI numbers
        run_number (int): Current run number
    """
    count_file_path = os.path.join(folder_path, 'data.csv')
    roi_number_lst_file_path = os.path.join(folder_path, 'roi_number_lst.npy')
    
    if run_number == 0 or run_number == 1:  # Initialize files for first run
        with open(count_file_path, 'w') as f_object:
            f_object.write(f'{survival_rate},{loop_var}\n')
        np.save(roi_number_lst_file_path, roi_number_lst)
    else:  # Append to existing files
        with open(count_file_path, 'a') as f_object:
            f_object.write(f'{survival_rate},{loop_var}\n')
        roi_number_lst_old = np.load(roi_number_lst_file_path)
        roi_number_lst_new = np.dstack((roi_number_lst_old, roi_number_lst))
        np.save(roi_number_lst_file_path, roi_number_lst_new)

def save_images(folder_path, first_image, second_image, run_number):
    """
    Save image data to files.
    
    Args:
        folder_path (str): Path to save the image files
        first_image (ndarray): First image data
        second_image (ndarray): Second image data
        run_number (int): Current run number
    """
    np.save(os.path.join(folder_path, 'first'), first_image)
    np.save(os.path.join(folder_path, 'seconds'), second_image)
    
    # Initialize or append to data file
    count_file_path = os.path.join(folder_path, 'data.csv')
    if run_number == 0:
        with open(count_file_path, 'w') as f_object:
            f_object.write('')
    else:
        with open(count_file_path, 'a') as f_object:
            f_object.write('')
