# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:44:31 2020

@author: Jacob
"""

import numpy as np

#TODO: replace local constants with imports from steck
from.steck_si import cesium


def numAtoms(Ncounts, ExpTime, Det,I=1):
    QE = 0.95 # photons/electron with our camera lens coating
    Gain = 100 #EM gain
    Sen = 18.6 #CCD sensitivity, electrons/count
    NA = 0.6 #numerical aperture of collection lens
#    I = 1 #intensity in units of Isat (imaging with spinpol beam: assume 3mm waist, 1.5mW incident power)
    Gamma = 2*np.pi*5.2227*10**6
    Rsc = 0.5*Gamma*I/(1+4*(Det/Gamma)**2+I) #Scattering rate
    
    SA=np.pi*(NA)**2 #solid angle collected by lens 
    
    Nphotons_cam=Ncounts*Sen*1/QE*1.0/Gain #convert camera counts to photons reaching the camera
    
    Nphotons=Nphotons_cam*(4*np.pi/SA) #Nphotons=actual number of photons scattered by the atoms
    
    Natoms= Nphotons*1/Rsc*1/ExpTime
    
    return Natoms

def fluorescenceNumAtoms(Ncounts, ExpTime, Det=0,I=1):
    QE = 0.55 # according to documents online about the Ixon Ultra (?)
    Gain = 100 #EM gain
    Sen = 18.6 #CCD sensitivity, electrons/count
    NA = 0.6 #numerical aperture of collection lens
#    I = 1 #intensity in units of Isat = 1.1 mW/cm^2  
    Gamma = 2*np.pi*5.2227*10**6
    Rsc = 0.5*Gamma*I/(1+4*(Det/Gamma)**2+I) #Scattering rate
    
    SA=np.pi*(NA)**2 #solid angle collected by lens 
    
    Nphotons_cam=Ncounts*Sen*1/QE*1.0/Gain #convert camera counts to photons reaching the camera
    
    Nphotons=Nphotons_cam*(4*np.pi/SA) #Nphotons=actual number of photons scattered by the atoms
    
    Natoms= Nphotons*1/Rsc*1/ExpTime
    
    return Natoms