# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:23:43 2020

@author: Jacob
"""

import scipy.constants as cnst

def J_to_unit(energy, unit):
    """
    Converts energy from J to specified unit.

    Parameters
    ----------
    energy : float
        Energy to convert in Joules.
    unit : string
        Unit to convert to.

    Returns
    -------
    energy : float
        Energy in specified unit.

    """
    
    #NOTE: JH 09/2022: Hz and MHz are actually rad/s and rad/us
    
    if unit == 'J':
        energy *= 1
    elif unit == 'Hz':
        energy *= 1/cnst.h
    elif unit == 'MHz':
        energy *= 1e-6/cnst.h
    elif unit == 'K':
        energy *= 1/cnst.Boltzmann
    elif unit == 'uK':
        energy *= 1e6/cnst.Boltzmann
    else:
        raise Exception('Unit must be J, Hz, MHz, K, or uK.')
        
    return energy


def K_to_unit(temperature, unit):
    Hz_per_K = 0.1309 * 1e12
    
    if unit == 'K':
        return temperature
    elif unit == 'uK':
        return temperature*1e6
    elif unit == 'Hz':
        return temperature * Hz_per_K
    elif unit == 'MHz':
        return temperature * Hz_per_K * 1e-6
    
def convert(qty, startunit, endunit):
    return #TODO: write this