# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:46:03 2019

@author: Quantum Engineer
"""

import h5py
import numpy as np
import os
import subprocess


def delete_and_repack(file, targets, verbose=False):
    """
    Delets the list of target groups/datasets and repacks the file to reclaim 
    disc space.

    Parameters
    ----------
    file : str
        file to clean up.
    targets : string or list of strings
        list of things to be deleted from the file.
    verbose : bool, optional
        If true, prints progress on deleting targets. The default is False.

    """
    delete(file, targets, verbose)
    repack(file)
    return

def delete(file, targets, verbose):
    if not isinstance(targets, list):
        targets = [targets]
        
    assert os.path.exists(file), 'File does not exist'
        
    with h5py.File(file, 'a') as f:
        for target in targets:            
            try:
                del f[target]
                if verbose:
                    print(f'deleted {target}')
                
            except:
                if verbose:
                    print(f'unable to delete {target}')
                pass
    

def repack(file):
    """
    Reclaims unused space in an h5 file, typically after deleting a group or
    dataset.
 
    Parameters
    ----------
    file : str
        Path to file to be repacked.

    Returns
    -------
    ret: int
        Return code for the subprocess.call command

    """
    
    directory = '\\'.join(file.split("\\")[:-1])
    temp = os.path.join(directory, 'temp.h5')
    
    CREATE_NO_WINDOW = 0x08000000
    ret = subprocess.call(fr"h5repack {file} {temp}", creationflags=CREATE_NO_WINDOW)
    os.remove(file)
    os.rename(temp, file)
    
    return ret


def findObject(group, objectname): 
    """
    Given a group and the name of an object, this function recursively searches
    through group/<> and returns the first match
    """
    
    def find_foo(name):
        if objectname in name:
            return name
        
    loc = group.visit(find_foo)
    
    return loc

def getDataset(target, path):
    if type(target) == h5py._hl.files.File:
        return target[path][:]
    
    with h5py.File(target, 'r') as f:
        return f[path][:]
        

def dictionaryToDatasets(group, dictionary, recursive=False, ignore=[]):
    """
    Given an h5 group and a dictionary, this function converts all keys, item
    in the dictionary into datasets containing the same item.
    """
    
    for key, item in dictionary.items():
        
        if not hasattr(item, "__len__"):
            item = [item]
        
        if (type(item) == dict) and recursive and not (item in ignore):
            newGroup = group.require_group(key)
            dictionaryToDatasets(newGroup, item, recursive=recursive)
            continue
        
        # convert bytes or unicode to str
        if type(item[0]) == bytes:
            for index, val in enumerate(item):
                item[index] = str(val)
        elif type(item[0]) == np.unicode_:
            item = item.astype(np.str_)

        
        if not key in group:
            if type(item[0]) in [str, np.str_]:
                dt = h5py.special_dtype(vlen=str)
            elif (type(item[0]) == bool):
                dt = bool
            elif type(item)==np.ndarray:
                dt = item.dtype
            else:
                dt = 'f'
                
            if type(item) == np.ndarray:
                shape = np.shape(item)
                maxshape = tuple([None]*item.ndim)
            else:
                shape = (len(item),)
                maxshape = (None,)
                
            group.create_dataset(key, shape=shape, dtype=dt, maxshape=maxshape)
                 
        else:
            pass #TODO: check to make sure dtype has not changed
            
        dataset = group[key]
        dataset.resize(len(item), axis=0) 
        
        #this doesn't work
        # if type(item) == np.ndarray:
        #     dataset.resize(np.shape(item))
        
        dataset[:] = item
            
    return group


def datasetsToDictionary(group, recursive=False, ignore=[]):
    """
    Given an h5 group, this function returns a dictionary with keys and items
    determined by the datasets present in the group. If the recursive
    boolean is true, this function returns a dictionary of dictionaries recursively
    generated from subfolders.
    """
    dictionary = dict()
    
    for name in group:
        item = group[name]
        
        if isinstance(item, h5py._hl.dataset.Dataset):
            dictionary[str(name)] = item[:]
        elif isinstance(item, h5py._hl.group.Group) and (recursive) and not (name in ignore):
            dictionary[str(name)] = datasetsToDictionary(group[name],
                                                         recursive=recursive,
                                                         ignore=ignore)
    
    return dictionary


def dictionaryToAttributes(hdf5_item, dictionary):
    for key, item in dictionary.items():
        hdf5_item.attrs[key] = item
        
        
def attributesToDictionary(group, recursive=False, ignore=[]):
    """
    Given an h5 group, this function returns a dictionary with keys and items
    determined by the attributes present in the group. If the recursive
    boolean is true, this function returns a dictionary of dictionaries recursively
    generated from subfolders.
    """
    dictionary = dict()
    
    for name in group:
        item = group[name].attrs
        dictionary[str(name)] = dict()
        
        for attrname in item:
            dictionary[str(name)][str(attrname)] = item[attrname]
     
        if (recursive) and not (name in ignore):
            dictionary[str(name)] = attributesToDictionary(group[name],
                                                         recursive=recursive,
                                                         ignore=ignore)
    return dictionary

def getAnalysisH5(directory, allow_multiple_files=False):
    """
    Returns an h5 file from a directory. By default raises an error if more
    than one h5 file is found because it presumes the file will be used for
    analysis.
    """
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.h5')]
    assert len(files) != 0, 'No possible analysis h5 files found'
    
    if not allow_multiple_files:
        assert len(files) < 2, 'More than one h5 file found to analyze'
        
    filename = files[0]
    print(f"Auto h5 found '{filename}'")
    
    return filename

def getGlobalsFromFile(file):
    """
    

    Parameters
    ----------
    file : str
        Path to h5 shot file.

    Returns
    -------
    globals_dict : dictionary
        Dictionary containing global : value of shot parameters.

    """
    
    with h5py.File(file, 'r') as f:
        globals_dict = attributesToDictionary(f)['globals']
    
    return globals_dict


#Functions originally from Rydberg Lab Autolyze

def getAttributeDict(group):
    """
    Returns a dictionary of attributes for <group> in an h5 file. Usage is:
        
        with h5py.File(shot, 'r') as f:
            attrDict = getAttributeDict(f['some group'])
    """
    
    if type(group) == str: #if we are passed a filepath
        group = h5py.File(group,'r')
    
    items = group.attrs.items()
    attributeDict = dict()
    
    for item in items:
        name = item[0]
        attributeDict[name] = item[1]
        
    return attributeDict