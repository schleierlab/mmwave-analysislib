# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:26:02 2019

@author: Quantum Engineer
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


def setROI(image, roi=None):
    if roi is None:
        return image
    else:
        imageROI = image[roi[1][0]:roi[1][1],roi[0][0]:roi[0][1]]
        return imageROI

def viewROI(imageROI): # for testing - to verify the correct region was chosen
    fig, ax = plt.subplots(1, 1) 
    ax.imshow(imageROI,cmap=plt.cm.jet, origin='lower')     
    return

def cropImageArray(imageArray, roi=None):
    if roi is None:
        return imageArray
    
    if imageArray.ndim == 2:
        imageArray = imageArray[roi[1][0]:roi[1][1],roi[0][0]:roi[0][1]]
    elif imageArray.ndim == 3:
        imageArray = imageArray[:, roi[1][0]:roi[1][1],roi[0][0]:roi[0][1]]
    else:
        raise Exception('ImageArray has wrong dimensions')
        
    
    return imageArray

def convexHullROI(rois, radius=0):
    
    # convert from tuple to list
    if type(rois[0]) == tuple:
        convert = True
        rois = [[list(pair) for pair in roi] for roi in rois]
    

    for idx, roi in enumerate(rois):
        if idx==0:
            ch_roi = roi
            continue
        
        for ii in range(2):
            for jj, extr in enumerate([min, max]):
                ch_roi[ii][jj] = extr([ch_roi[ii][jj], roi[ii][jj]])
                
    if radius != 0:
        ch_roi[0][0] += -radius
        ch_roi[0][1] += radius
        ch_roi[1][0] += -radius
        ch_roi[1][1] += radius
                
    if convert:
        ch_roi = tuple([tuple(pair) for pair in ch_roi])
        
    return ch_roi


def showROIPreview(image,previewROI,rois,title='',roiLabel='',filename='roiPreview.png', show_plot=True):
    '''
    

    Parameters
    ----------
    image : ndarray
        Image to be previewed.
    previewROI : roi tuple
        Roi to be previewed.
    rois : dict
        Dictionary of sub rois, usually denoting tweezers. Keys are used as roi names
        printed on the plot.
    title : string, optional
        Plot title. The default is ''.
    roiLabel : string, optional
        Label to be appended for each roi in rois. The ending is "_{roiIdx}". The default is ''.
    filename : string, optional
        Plot filename. The default is 'roiPreview.png'.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(figsize=(10,4))
    previewimg_cropped = setROI(image, previewROI)

    im = ax.imshow(previewimg_cropped, 
                   extent=[previewROI[0][0]-0.5,previewROI[0][1]+0.5,
                           previewROI[1][1]-0.5, previewROI[1][0]+0.5],
                   vmin=0, 
                   vmax=np.max(previewimg_cropped))
    
    ax.set_title(title)    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im,cax=cax)
    
    for idx, roi in rois.items():
        rect = patches.Rectangle((roi[0][0],roi[1][0]),roi[0][1]-roi[0][0],roi[1][1]-roi[1][0] ,linewidth=1,edgecolor='w',facecolor='none')
        ax.add_patch(rect)
        ax.text(roi[0][0],roi[1][0]-1.5, roiLabel+f'_{idx}',color='C1')
        
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

def selectRectangleROI(image):
    """
    Opens a plot and prompts the user to select a rectangular ROI by clicking
    on two opposite edges of the ROI.

    Parameters
    ----------
    image : 2D numpy array
        image over with the user selects a roi.

    Returns
    -------
    coords : TYPE
        list of two rectangle edge tuples.

    """
    get_ipython().run_line_magic('matplotlib', 'qt')

    fig, ax = plt.subplots()
    ax.imshow(image)
    
    plt.show()
    coords = []
    
    def addCoordinate(event, coords):
        ix = int(event.xdata)
        iy = int(event.ydata)
        
        print('x = {}, y = {}'.format(ix,iy))
    
        coords.append((ix, iy))
        
        print(len(coords))
    
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    
    cid = fig.canvas.mpl_connect('button_press_event',
                                 lambda event: addCoordinate(event, coords))
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    return coords

def separateROIs(rois,exclude=None):
    '''
    Take a dict of ROIs saved by e.g. extractROI.py and splits them into separate
    "areas" corresponding to each tweezer. Each area contains an F3 and an F4
    roi name. The way we save roi names is likely to change so this function
    will need to change too.

    Parameters
    ----------
    rois : dict
        Dictionary of ROI names.
    exclude : string, optional
        Exclude an area if the roi contains "exclude". The default is None.

    Returns
    -------
    splitROI : dict
        Dictionary of areas.

    '''
    splitROI={}
    for key, roi in rois.items():
        if exclude:
            if exclude in key:
                continue
        #make a dictionary of distinct areas to analyze where each area contains
        #one F=4 and one F=3 roi
        roiName = key.split('_')[0][:-1]
        try:
            roiIdx = key.split('_')[1]
        except IndexError:
            roiIdx = ''
        roiF = key.split('_')[0][-1]
        
        if roiName+roiIdx not in splitROI:
            splitROI[roiName+roiIdx]={}
            splitROI[roiName+roiIdx]['F'+roiF] = key
        else:
            splitROI[roiName+roiIdx]['F'+roiF] = key
            
    return splitROI

if __name__ == '__main__':
    imageFP = r'C:\Users\Jacob\Documents\GitHub\tweezerlyze\testing\atoms_filled.npy'
    image = np.load(imageFP)

    coordinates = selectRectangleROI(image)