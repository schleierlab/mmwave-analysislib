# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:52:06 2020

@author: Jacob
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage.interpolation import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from . import andorlyze
from ..data import autolyze as az
from ..data import read as rf
from .roi import setROI
from .process import getMean

def getImage(shot, camera):
    with h5py.File(shot,'r') as f:
        if 'images' in f.keys():
            if camera in f['images']:
                data = list(f['images'][camera].items())[0]
                image = np.array(data[1]) #TODO: generalize this for multiple images per shot
            else:
                print(f'No {camera} pictures in file {shot}. Make sure the camera name matches h5/images/camera')
        else:
            print(f'No images in file {shot}')
    return image


def extractImages(target, camera='andor', background_fp=None, roi=None, doMean=False,
               doSingle=True, printImages=True, saveImages=True, zoomRatio=1, 
               markers=[], vmin=None, vmax=None, demask=True):
    
    fp = az.getDirectory(target)
    
    filepaths, filenames=rf.open_all_HDF5_in_dir(fp)
    
    if background_fp is not None:
        if background_fp.endswith('.npy'):
            background = np.load(background_fp)
        elif background_fp.endswith('h5'):
            background = rf.getdata(background_fp, 'atoms1')
        else:
            raise Exception('Invalid background image path.')
    else:
        background = None

    
    if doMean:
        meanimgs = getMean(fp)
        
        if background is not None:
            meanimgs = meanimgs - background
            
        if demask and camera=='andor':
            meanimgs = andorlyze.demaskImage(meanimgs, background)
        
        meanimgs = setROI(meanimgs,roi) 
    
        fig, ax = plt.subplots() 
        im = ax.imshow(meanimgs, cmap='viridis')#, vmax=50)		
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im,cax=cax)
        
        save_fp = os.path.join(fp, fp.split('\\')[-1])
        fig.savefig(save_fp+'mean.png', dpi=500, bbox_inches='tight')
        
        if not printImages:
            plt.close(fig)
        
        np.save(save_fp+'mean.npy', meanimgs)
       
        
    if doSingle:
        for index1, item in enumerate(filepaths): 
                with h5py.File(item,mode='r+') as f:
                    if camera == 'manta145': #len(f['devices']['AVTManta145'])>0:
                        
                        numFrames = f['devices']['AVTManta145']['EXPOSURES'].shape[0] 
                        for index2 in range(numFrames):
                            name = 'atoms' + str(index2)
                            data = rf.getdata(item, name)
                            
                            imageROI = setROI(data,roi) 
                            
                            if zoomRatio != 1:
                                imageROI = zoom(imageROI, zoomRatio)
                                
                            fig, ax = plt.subplots() 
                            im = ax.imshow(imageROI, cmap='magma', vmin=vmin, vmax=vmax)		
                            
                            for markidx, marker in enumerate(markers):
                                ax.scatter(marker[0],marker[1],s=5)
                            
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(im,cax=cax)
                            
                            filename =  str(filenames[index1])[0:-3] + "_" + name + ".png"
                            save_fp = os.path.join(fp, filename)
                            fig.savefig(save_fp,bbox_inches='tight',dpi=200)
                            
                            plt.close(fig)	
        
                    
                    if camera == 'manta223': # len(f['devices']['AVTManta223'])>0:
                        numFrames = f['devices']['AVTManta223']['EXPOSURES'].shape[0] 
                        roi = ((0,2048),(0,1088))
                        
                        for index2 in range(numFrames):
                            name = 'atoms' + str(index2)
                            data=rf.getdata(item, name) #FIXME: make consistent with h5 structure
                            
                            imageROI = setROI(data,roi) 
                            if zoomRatio != 1:
                                imageROI = zoom(imageROI, zoomRatio)
                            fig, ax = plt.subplots() 
                            im = ax.imshow(imageROI, cmap='jet')#, vmax=4096)		
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(im,cax=cax)
                            fig.savefig(fp+"/"+str(filenames[index1])[0:-3]+"_"+name+".png",bbox_inches='tight',dpi=200)
                            plt.close(fig)	
                        
                                       
                    if camera == 'andor': #len(f['devices']['AndorIxon'])>0:
                        numFrames = f['devices']['AndorIxon']['EXPOSURES'].shape[0] 
                        for index2 in range(numFrames):
                            
                            data = andorlyze.getImage(item)
                            
                            if demask:
                                data = andorlyze.demaskImage(data, background)
                            
                            imageROI = setROI(data,roi) 
                            if zoomRatio != 1:
                                imageROI = zoom(imageROI, zoomRatio)
                            fig, ax = plt.subplots() 
                            im = ax.imshow(imageROI, cmap='viridis',vmin=None, vmax=None)		
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            plt.colorbar(im,cax=cax)

                            name = 'atoms_andor_' + str(index2)
                            fig.savefig(fp+"/"+str(filenames[index1])[0:-3]+"_"+name+".png",bbox_inches='tight',dpi=200)
                            plt.close(fig)     