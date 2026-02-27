# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:08:51 2020

@author: Jacob
"""

import numpy as np
import scipy.constants as cnst
from .steck_si import cesium
from .intensity import saturationIntensity, peakIntensity
from .unit_conversions import J_to_unit


def depthGrimmClassical(species=cesium, wavelength=1064e-9, power=None, waist=None,
          intensity=None, verbose=False, unit='Hz'):
    """
    Calculates trap depth following equation 10 in OPTICAL DIPOLE TRAPS FOR 
    NEUTRAL ATOMS by Grimm and Weidemuller.

    Parameters
    ----------
    species : Atom class, optional
        Class containing atomic properties, as defined in steck.py. The default
        is cesium.
    wavelength : float, optional
        Trapping beam wavelength in m. The default is 1064e-9.
    power : float, optional
        Trapping beam power in W. The default is None.
    waist : float, optional
        Trapping beam waist in m. The default is None.
    intensity : float, optional
        Trapping beam intensity in W/m2. The default is None.
    verbose : bool, optional
        Set to True to display debugging values. The default is False.
    unit : str, optional
        Unit for the returned trap depth, can be J, K, Hz or MHz. The default is 'Hz'.

    Returns
    -------
    depth : float
        Trap depth in specified units.

    """
    
    # frequency of drive field in Hz
    drive_frequency = cnst.c/wavelength
    # and rad/s
    omega_drive = 2*np.pi*drive_frequency
    
    # intensity in W/m2
    if intensity is None:
        I = peakIntensity(power, waist)
    else:
        I = intensity
    
    # classical depth for each transition
    def u2level(omega_transition, linewidth):
        prefactor = (3*np.pi*cnst.c**2)/(2*omega_transition**3)
        freqfactor = -linewidth*(1/(omega_transition-omega_drive) + 1/(omega_transition+omega_drive))
        return prefactor*freqfactor*I
    
    # full depth is weighted by transition
    depth = (2/3)*u2level(2*np.pi*species.D2.frequency, species.D2.linewidth) \
            + (1/3)*u2level(2*np.pi*species.D1.frequency, species.D1.linewidth)
        
    # change energy unit
    depth = J_to_unit(depth, unit)
    
    if verbose:
        print('intensity', I)
        print('depth:', depth, unit)
        print('')
        
    return -depth


def depthSteckClassical(species=cesium, wavelength=1064e-9, power=None,
                          waist=None, intensity=None, verbose=False, unit='Hz'):
    """
    Calculates trap depth following equation 1.76 in Quantum and Atom Optics
    by Steck. This method looks up the saturation intensity for D1 and D2.

    Parameters
    ----------
    species : Atom class, optional
        Class containing atomic properties, as defined in steck.py. The default
        is cesium.
    wavelength : float, optional
        Trapping beam wavelength in m. The default is 1064e-9.
    power : float, optional
        Trapping beam power in W. The default is None.
    waist : float, optional
        Trapping beam waist in m. The default is None.
    intensity : float, optional
        Trapping beam intensity in W/m2. The default is None.
    verbose : bool, optional
        Set to True to display debugging values. The default is False.
    unit : str, optional
        Unit for the returned trap depth, can be J, K, Hz or MHz. The default is 'Hz'.

    Returns
    -------
    depth : float
        Trap depth in specified units.

    """
    
    # frequency of laser field in Hz
    drive_frequency = cnst.c/wavelength
    # and rad/s
    omega_drive = 2*np.pi*drive_frequency
    
    # intensity in W/m2
    if intensity is None:
        I = peakIntensity(power, waist)
    else:
        I = intensity
    
    # classical depth for each transition
    def lineDepth(omega_transition, linewidth, I_sat):
        prefactor = cnst.hbar * linewidth**2 / 8
        freqfactor = 1/(omega_drive - omega_transition) - 1/(omega_drive + omega_transition)
        s0 = I/I_sat
        return prefactor*freqfactor*s0
    
    
    # total depth is summed over transitions    
    lines = ['D2', 'D1']
    depth = 0
    for line in lines:           
        t = getattr(species, line)
        I_sat = saturationIntensity(species, line, far_detuned=True)
        depth += lineDepth(2*np.pi*t.frequency, t.linewidth, I_sat)
        
    # change energy unit
    depth = J_to_unit(depth, unit)
            
    if verbose:
        print('intensity:', I)
        print('depth:', depth, unit)
        print('')
        
    return -depth
        

def depthSteckQuantum(species=cesium, wavelength=1064e-9, power=None, waist=None, 
                        intensity=None, hyperfine_offset=0, verbose=False, unit='Hz', method='simplified'):
    """
    Calculates trap depth following equations 7.304 and 7.457 in Quantum and
    Atom Optics by Steck.

    Parameters
    ----------
    species : Atom class, optional
        Class containing atomic properties, as defined in steck.py. The default
        is cesium.
    wavelength : float, optional
        Trapping beam wavelength in m. The default is 1064e-9.
    power : float, optional
        Trapping beam power in W. The default is None.
    waist : float, optional
        Trapping beam waist in m. The default is None.
    intensity : float, optional
        Trapping beam intensity in W/m2. The default is None.
    verbose : bool, optional
        Set to True to display debugging values. The default is False.
    unit : str, optional
        Unit for the returned trap depth, can be J, K, Hz or MHz. The default is 'Hz'.
    method : str, optional
        Method for calculating stark shift, can be 'full' or 'simplified'. Should
        be equivalent.

    Returns
    -------
    depth : float
        Trap depth in specified units.
        
    """   

    # Steck quantum optics 7.296 connects reduced matrix element with linewidth
    def rMatrixElement(omega_transition, linewidth, Jg, Je):
        prefactor = 3*np.pi*cnst.epsilon_0*cnst.hbar*cnst.c**3
        degeneracyfactor = (2*Je+1)/(2*Jg+1)
        return np.sqrt(degeneracyfactor*prefactor*linewidth/omega_transition**3)
    
    # Steck quantum optics 7.485 (update: 7.457?) gives formula for lare detuning optical stark shift
    def OpticalStarkShiftFull(omega_transition, omega_drive, linewidth, Jg, Je, I):
        #In steck this is the positive frequency E field component E_0+ for which |E_0|^2 = 4 |E_0+|^2:
        Esquared = I*2*cnst.mu_0*cnst.c 
        freqfactor = omega_transition/(omega_transition**2-omega_drive**2)
        return -2*freqfactor*Esquared*rMatrixElement(omega_transition, linewidth, Jg, Je)**2/(3*cnst.hbar)   # NOTE the 3 is back
    
    # simplified formula
    def OpticalStarkShiftSimplified(omega_transition, omega_drive, linewidth, Jg, Je, I):
        degeneracyfactor = (2*Je+1)/(2*Jg+1)
        freqfactor = omega_transition**2*(omega_transition**2-omega_drive**2)
        return -1*degeneracyfactor*np.pi*cnst.c**2*linewidth*I/freqfactor   # NOTE (09/22) added a factor of 12 to match the full shift
    
    # choose calculation method (should be equivalent)
    if method =='simplified':
        f = OpticalStarkShiftSimplified
    elif method == 'full':
        f =  OpticalStarkShiftFull
    else:
        raise Exception('Method must be simplified or full.')
        
    # intensity in W/m2
    if intensity is None:
        I = peakIntensity(power, waist)
    else:
        I = intensity
        
    # frequency of laser field in Hz
    drive_frequency = cnst.c/wavelength
    # and rad/s
    omega_drive = 2*np.pi*drive_frequency
    
    # calculate depth for each transition
    depths = {}
    transitions = ['D2', 'D1']
    for transition in transitions:
        t = getattr(species, transition)
        
        omega_transition = 2*np.pi*(t.frequency - hyperfine_offset) 
        linewidth = t.linewidth
        Jg = t.Jg
        Je = t.Je
        
        if verbose:
            print('omega_transition:', omega_transition)
            print('omega_drive:', omega_drive)
            print('linewidth', linewidth)
            print('Jg', Jg)
            print('Je', Je)
            print('I:', I)
            print('')
        
        depths[transition] = f(omega_transition, omega_drive, linewidth, Jg, Je, I)
        
    depth = np.sum(list(depths.values())) #J

    # change energy unit
    depth = J_to_unit(depth, unit) #NOTE: JH 9/22, Hz and MHz are actually rad/s and rad/us
        
    if verbose:
        print('depth:', depth)
        
    return -depth


def trapDepth(power=None, waist=None, intensity=None, method='steck_quantum', **kwargs):
    if not ((power and waist) or (intensity)):
        raise Exception('Must provide power and waist or intensity.')
        
    if method == 'grimm_classical':
        return depthGrimmClassical(power=power, waist=waist, intensity=intensity, **kwargs)
    elif method == 'steck_classical':
        return depthSteckClassical(power=power, waist=waist, intensity=intensity, **kwargs)
    elif method == 'steck_quantum':
        return depthSteckQuantum(power=power, waist=waist, intensity=intensity, **kwargs)
    else:
        raise Exception('Invalid method')
        
        
def hyperfineTrapDepth(power=None, waist=None):
    f4offset = 4.02e9
    f3offset = 5.17e9
    
    f4 = depthSteckQuantum(species=cesium, wavelength=1064e-9, power=power, waist=waist, hyperfine_offset = f4offset,
                        intensity=None, verbose=False, unit='Hz', method='simplified')
    
    f3 = depthSteckQuantum(species=cesium, wavelength=1064e-9, power=power, waist=waist, hyperfine_offset = -f3offset,
                        intensity=None, verbose=False, unit='Hz', method='simplified')  
    
    return f4 - f3
