# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:23:18 2020

@author: Jacob
"""

import numpy as np
import scipy.constants as cnst
from .steck_si import cesium


def peakIntensity(power, waist):
    """
    Returns intensity for given power (W) and waist (m).

    Parameters
    ----------
    power : float
        Power in W. The default is None.
    waist : float
        Beam wasit in m. The default is None.

    Returns
    -------
    float
        Intensity in w/m2.

    """
    
    return 2*power/(np.pi*(waist**2))

def fieldToIntensity(field_amplitude):
    return 0.5 * cnst.c * cnst.epsilon_0 * field_amplitude**2

def intensityToField(intensity):
    return np.sqrt(2*intensity/(cnst.c * cnst.epsilon_0))


def saturationIntensity(species=cesium, line='D2', F=4, Fprime=5, 
                        polarization='sigma_plus', configuration='pumped_plus',
                        far_detuned=False, debug=False):
    
    # simple case: large detuning uses characteristic dme (steck 7.497)
    if far_detuned:
        dmefactor = 1/3
        
    # harder case: must count up every hyperfine transition
    else:            
        # restrict mF values depending on distribution in manifold {mF: weight}
        if type(configuration) == dict:
            mF_dict = configuration
        elif configuration == 'clock':
            mF_dict = {0: 1}
        elif configuration == 'pumped_plus':
            mF_dict = {F: 1}
        elif configuration == 'pumped_minus':
            mF_dict = {-F: 1}
        elif configuration == 'distributed':
            mF_dict = {mF: 1/(2*F+1) for mF in range(-F, F+1)}
        else:
            raise Exception ('Configuration must be a dict of mF levels with weights or a string: clock, pumped_plus, pumped_minus, or even.')
    
        # mF shift for debugging
        mF_shift = {
            'sigma_minus': -1,
            'pi': 0,
            'sigma_plus': +1,
            }
    
        # keep track of dmefactor for each hyperfine transition {mF: [channel strengths]}
        dmefactor_dict = {mF:[] for mF in range(-F, F+1)}
        
        # keys are mF values
        dme_coeffs = species.dme_coeffs[line][F][polarization][Fprime]
        
        # add up contributions weighted across mF transitions        
        for mF in mF_dict.keys():            
            # make sure transition exists
            mFprime = mF + mF_shift[polarization]
            
            if np.abs(mFprime) > Fprime:
                if debug:
                    print(f'F={F}, mF={mF}  to  Fprime={Fprime}, mFprime={mFprime}  does not exist (skipping)')
                continue
            elif debug:
                print(f'F={F}, mF={mF}  to  Fprime={Fprime}, mFprime={mFprime}')
            
            dme_coeff = dme_coeffs[mF]
            dmefactor_dict[mF].append(dme_coeff**2)
            
        # average across transitions for each mF
        for mF, val in dmefactor_dict.items():
            if len(val) > 0:
                dmefactor_dict[mF] = np.mean(val)
            else:
                dmefactor_dict[mF] = 0
                
        # weight by relative population
        dmefactor = np.sum([weight*dmefactor_dict[mF] for mF, weight in mF_dict.items()])
        
    # rescale by actual dme
    dme = getattr(species, line).dme
    dmefactor *= (dme**2)
    
    #Steck 5.249
    linewidth = getattr(species, line).linewidth
    prefactor = cnst.c * cnst.epsilon_0 * linewidth**2 * cnst.hbar**2 / (4)
    Isat = prefactor/dmefactor
    
    if debug:
        print('Isat', Isat/10, 'mw/cm2') #mW/cm2
    
    return Isat