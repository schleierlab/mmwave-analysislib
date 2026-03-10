# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:47:42 2016

@author: Stanford University
"""

import numpy as np
from os import listdir,path
from re import compile
import h5py
# from astropy.io import fits

"""
NOTE: this file contains old cavity lab functions, which we've kept only for
limited backwards compatability. New code should use analysis.data.autolyze
"""

def open_all_HDF5_in_dir(directory_name):
    list_of_files=listdir(directory_name)
    re_sis=compile('\.h5')
    file_path_array=[]
    file_name_array=[]
    for item in list_of_files:
        if re_sis.search(item,1):
            file_path_array.append(path.join(directory_name,item))
            file_name_array.append(item)
    return file_path_array, file_name_array


def get_all_fitsfiles(directory_name):
    list_of_files=listdir(directory_name)
    re_sis=compile('\.fits')
    file_path_array=[]
    file_name_array=[]
    for item in list_of_files:
        if re_sis.search(item,1):
            file_path_array.append(path.join(directory_name,item))
            file_name_array.append(item)
    return file_path_array, file_name_array
    


def read_fits(filename):
    fitsdat=fits.open(filename)
    image=fitsdat[0].data
    return image[0]
    

def getdata(path, dataname): 
#path=complete path including filename.h5, dataname=variable name where data is stored, like "MOT3D_Fluorescence" or "flat"
    def find_foo(name):
        if dataname in name:
            return name
    with h5py.File(path,'r') as hf:
        loc=hf.visit(find_foo)
        if loc is None:
            print("Warning: no match found")
        dat=np.array(hf.get(loc))
        return dat


def getxval(path, global_group, global_name): 
#path to HDF5 including filename.h5, global_group specifies the global variable we are iterating with, e.g., 'MOT', global_name specifies the global variable we are iterating with, e.g., 'MOT_LoadTime'
    with h5py.File(path,'r') as hf:
        globalgroup=hf.get('globals/'+global_group)
        xval=globalgroup.attrs[global_name]
        return xval
        
    
def getxval2(path, global_name): 
    raise Exception('Deprecated. Use analysis.data.autolyze.getParam')
#path to HDF5 including filename.h5, global_name specifies the global variable we are iterating with, e.g., 'MOT_LoadTime'
    with h5py.File(path,'r') as hf:
        globalgroup=hf.get('globals')
        xval=globalgroup.attrs[global_name]
        return xval