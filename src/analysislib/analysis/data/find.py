# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:06:14 2022

@author: RYD-WINDTUNNEL
"""

import os, fnmatch

def find(directory, find, file_pattern):
    
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, file_pattern):
            # prepare filepaths
            filepath = os.path.join(path, filename)
            
            # read data
            with open(filepath) as f:
                try:
                    s = f.read()
                except Exception as e:
                    print(f'Error while reading file {filepath}')
                    raise Exception(e)
            
            # find
            if find in s:
                # print(f'found {find} in {filepath}')
                print(filepath)
                
directory = os.getcwd()
file_pattern = '*.py'


find(directory, 'compare', file_pattern)