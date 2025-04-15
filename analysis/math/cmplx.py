# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:52:41 2020

@author: Jacob
"""

import numpy as np

def extractComplexPhase(phase, A, offset):
    '''
    A function which extracts a phase from a complex number A*exp**(phase+offset).
    Phase is in degrees.
    '''
    return np.angle(A*np.exp(1j*(phase*np.pi/180 + offset)))%(2*np.pi)