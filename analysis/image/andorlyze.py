# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:04:43 2018

@author: Quantum Engineer
"""

import h5py
import numpy as np
import os

from analysis.data import autolyze as az
from analysis.image.process import getMean

def getImage(shot):
    with h5py.File(shot,'r') as f:
        if 'images' in f.keys():
            if 'andor' in f['images']:
                data = list(f['images']['andor'].items())[0]
                image = np.array(data[1])
            else:
                print('No Andor pictures in file ' + shot)
        else:
            print('No pictures in file ' + shot)
            
    return image


    

def getImageArray(shots, transposeImage=False, excludeBlanks=False, demask=False):
    """
    Given an array of shot filepaths, returns an array of the first Andor image
    in each file
    """
    
    image_array = []
    
    for shot in shots:
        image = getImage(shot)
        
        if np.max(image) == 0:
            print(f'Warning: zero image found in shot {shot}')
            if excludeBlanks:
                    continue
        
        if transposeImage:
            image = image.transpose()
        
        image_array.append(image)

    return np.array(image_array) #IF SOMETHING BROKE, IT'S PROLLY 'CAUSE YOU'RE RUNNING 32-BIT PYTHON ON EXPT-CTRL- RUN ON ANALYSIS-1


def demaskImage(image, background, mask=None):
    """
    
    Parameters
    ----------
    image : 2d numpy array
        image to be demasked.
    background : str or 2d numpy array
        either the filepath or the actual background array.
    mask : str or 2d numpy array
        either the filepath or the actual background array.

    Returns
    -------
    image_demasked : 2d numpy array
        demasked image.

    """
    
    if mask is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        mask = os.path.join(dir_path, 'masks/mask_20200929.npy')
        
    if type(mask)==str:
        mask = np.load(mask)
    
    if type(background)==str:
        background = np.load(background)
    
    # extract signal
    signal = image-background
    
    # demask
    signal_demasked = signal/mask
    
    # reconstruct image
    image_demasked = signal_demasked + background
    return image_demasked


def demaskImageArray(image_array, background=None, mask=None):
    demasked_image_array = []
    
    if type(image_array) == list:
        returnAsList = True
    else:
        returnAsList = False
    
    if mask is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        mask = os.path.join(dir_path, 'mask.npy')
        
    if type(mask)==str:
        mask = np.load(mask)
    
    if type(background)==str:
        background = np.load(background)
    
    for image in image_array:
        image_demasked = demaskImage(image, background, mask)
        demasked_image_array.append(image_demasked)
        
    if not returnAsList:
        demasked_image_array = np.array(demasked_image_array)
    
    return demasked_image_array


def generateMask(fp, backgroundfp, fk_number=1, minimum=0.1):
    shots = az.getShots(fp)
    images = getImageArray(shots, demask=False)
    images = np.array(images)
    mask = np.mean(images, axis=0)

    background = np.load(backgroundfp)
    
    mask = mask-background
    mask = mask/np.max(mask)
    
    if fk_number > 1:
        shape = np.shape(mask)
        fk_y = shape[0]//fk_number
        mask = mask[:fk_y, ...]
        mask = np.tile(mask, (fk_number,1))
        
    padrows = 1024 - len(mask)
    if padrows>0:
        padding = np.ones((padrows, 1024))
        mask = np.concatenate([mask, padding], axis=0)
        
    mask[mask<minimum]=minimum
    
    dir_path = os.path.dirname(os.path.realpath(__file__))    
    mask_fp = os.path.join(dir_path, 'mask.npy')
    
    np.save(mask_fp, mask)
    
    return mask

def getImageOld(shot):
    with h5py.File(shot,'r') as f:
        if 'AndorIxonImages' in f.keys():
            data = list(f['AndorIxonImages'].items())[0]
            image = np.array(data[1])
        else:
            print('No pictures in file ' + shot)
            
    return image

def getImageArrayOld(shots, transposeImage=False, excludeBlanks=False, demask=False):
    """
    Given an array of shot filepaths, returns an array of the first Andor image
    in each file
    """
    
    image_array = []
    
    for shot in shots:
        image = getImageOld(shot)
        
        if np.max(image) == 0:
            print(f'Warning: zero image found in shot {shot}')
            if excludeBlanks:
                    continue
        
        if transposeImage:
            image = image.transpose()
        
        image_array.append(image)

    return np.array(image_array)
    
if __name__ == '__main__':
    fp = r'Z:\Experiments\rydberglab\Tweezers\2020\09\29\193233__C_repeats__forDemasking'
    backgroundfp = r'\\RYD-EXPTCTRL\labscript_userlib\analysislib\analysis\image\backgrounds\andor\fluorescence\dark10MHz\dark_noFK_10MHz.npy'

    mask = generateMask(fp, backgroundfp, fk_number=3)