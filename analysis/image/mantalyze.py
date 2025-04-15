# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:31:23 2019

@author: Quantum Engineer
"""

import h5py
import numpy as np

def getImage(shot, idx=0):
    with h5py.File(shot,'r') as f:
        if 'images' in f.keys():
            data = list(f['images'].items())
            imageDataTypes = [ii[0] for ii in data]
            if 'andor' in imageDataTypes:
                data.pop(imageDataTypes.index('andor'))
            try:
                image = np.array(data[idx][1])
            except IndexError:
                raise IndexError('Image index out of range.')
        else:
            print('No pictures in file ' + shot)
        
    return image

def getImages(shot):
    images = []
    with h5py.File(shot,'r') as f:
        if 'images' in f.keys():
            for item in f['images'].items():
                try:
                    image = np.array(item[1])
                    assert len(image) > 1, 'Probably trying to grab an andor image, moving on'
                    images.append(image)
                except Exception as e:
                    print(e)
                    continue
        else:
            print('No pictures in file ' + shot)
        
    return np.array(images)

def getCameraInfo(shot):
    infoDict = {}
    with h5py.File(shot,'r') as f:
        for key in f['devices']:
            if 'manta' in key.lower():
                infoDict[key] = {}
                item = f['devices'][key].attrs
                for attr in item:
                    infoDict[key][str(attr)] = item[attr]
                
    return infoDict


def getImageArray(shots, idx=0, transposeImage=False, excludeBlanks=False):
    """
    Given an array of shot filepaths, returns an array of the first Manta image
    in each file
    """
    
    image_array = []
    
    for shot in shots:
        image = getImage(shot,idx=idx)
        
        if np.max(image) == 0:
            print(f'Warning: zero image found in shot {shot}')
            if excludeBlanks:
                    continue
        
        if transposeImage:
            image = image.transpose()
        
        image_array.append(image)

    return np.array(image_array) #IF SOMETHING BROKE, IT'S PROLLY 'CAUSE YOU'RE RUNNING 32-BIT PYTHON ON EXPT-CTRL- RUN ON ANALYSIS-1