# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 13:40:58 2018

@author: Quantum Engineer
"""

import os
import imageio
from ..data import read as rf
import datetime
import matplotlib.pyplot as plt


def imagesToGif(directory, filename):
    if not filename.endswith('.gif'):
        filename = f'{filename}.gif'
    
    filepaths, filenames=rf.open_all_HDF5_in_dir(directory)
    
    for myfile in filenames:
    	print(myfile)
    	
    for mypath in filepaths:
    	print(mypath)
    
    imgs=[]
    
    for item in filepaths:
        data=rf.getdata(item, "testdataset")
        imgs.append(data)
    	
    filepath = os.path.join(directory, filename)
    imageio.mimsave(filepath, imgs)
    
    return

def savePlotData(data, fp, filename):
    """
    Given a list data containing all of the vectors in a plot, this function 
    writes each as a line in the csv file fp/filename.csv
    """
    
    txtfile = open(os.path.join(fp, filename+'.csv'), 'a+')
    
    for myline in data:
        mystr = ''
        
        for index, val in enumerate(myline):
            if not index==0:
                mystr = mystr + ', '
            mystr = mystr + str(val)
        
        txtfile.write(mystr + '\n')
        
    txtfile.close()
    
def dateString(d=None, m=None, y=None):
    if d is None:
        d = datetime.datetime.today().strftime('%d')
    if m is None:
        m = datetime.datetime.today().strftime('%m')
    if y is None:
        y = datetime.datetime.today().strftime('%Y')
        
    return f'{y}{m}{d}'

def savefig(fp, fig=None, bbox_inches='tight', dpi=500, temp_location=r'Z:\Experiments\rydberglab', **kwargs):    
    if fig is None:
        fig = plt.gcf()
    
    try:
        fig.savefig(fp, bbox_inches=bbox_inches, dpi=dpi)
        
    except FileNotFoundError:
        # saving a figure with fp longer than 259 characters results in a file not found error
        print('Warning: savefig filepath is too long. Prepending with \\?\.')
        fp = r"\\\\?\\" + fp
        fig.savefig(fp, bbox_inches=bbox_inches, dpi=dpi)