# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:52:48 2020

@author: Jacob

Cesium properties pulled from D Steck, Cesium D Line Data
"""

import numpy as np
from numpy import sqrt as s


class quantity(float):
    def __new__(self, value, unit):
        return float.__new__(self, value)
    def __init__(self, value, unit):
        float.__init__(value)
        self.unit = unit

class Transition():
    def __init__(self, properties):
        for k, qtuple in properties.items():
            value, unit = qtuple
            setattr(self, k, quantity(value, unit))


class Atom():
    def __init__(self, properties, transition_properties):
        
        for key, qtuple in properties.items():
            value, unit = qtuple
            setattr(self, key, quantity(value, unit))
        
        for line, tp in transition_properties.items():
            setattr(self, line, Transition(tp))
            
    def getDME(self, line, polarization, F, mF, Fprime):
        coeff = self.dme_coeffs[line][F][polarization][Fprime][mF] 
        transition = getattr(self, line)
        return coeff * transition.dme


# line / F / polarization / Fprime / mF
dme_coeffs = {
    'D2': {
           4: {
               'sigma_plus': {
                   5: {
                       -4: s(1/90),
                       -3: s(1/30),
                       -2: s(1/15),
                       -1: s(1/9),
                       0: s(1/6),
                       1: s(7/30),
                       2: s(14/45),
                       3: s(2/5),
                       4: s(1/2),
                       },
                  4: {
                       -4: s(7/120),
                       -3: s(49/480),
                       -2: s(21/160),
                       -1: s(7/48),
                       0: s(7/48),
                       1: s(21/160),
                       2: s(49/480),
                       3: s(7/120),
                       },
                   3: {
                       -4: s(7/72),
                       -3: s(7/96),
                       -2: s(5/96),
                       -1: s(5/144),
                       0: s(1/48),
                       1: s(1/96),
                       2: s(1/288),
                       },
                   },
               'pi': {
                   5: {
                       -4: -s(1/10),
                       -3: -s(8/45),
                       -2: -s(7/30),
                       -1: -s(4/15),
                       0: -s(5/18),
                       1: -s(4/15),
                       2: -s(7/30),
                       3: -s(8/45),
                       4: -s(1/10),
                       },
                  4: {
                       -4: -s(7/30),
                       -3: -s(21/160),
                       -2: -s(7/120),
                       -1: -s(7/480),
                       0: s(0),
                       1: s(7/480),
                       2: s(7/120),
                       3: s(21/160),
                       4: s(7/30),
                       },
                   3: {
                       -3: s(7/288),
                       -2: s(1/24),
                       -1: s(5/96),
                       0: s(1/18),
                       1: s(5/96),
                       2: s(1/24),
                       3: s(7/288),
                       },
                   },
               'sigma_minus': {
                   5: {
                       -4: s(1/2),
                       -3: s(2/5),
                       -2: s(14/45),
                       -1: s(7/30),
                       0: s(1/6),
                       1: s(1/9),
                       2: s(1/15),
                       3: s(1/30),
                       4: s(1/90),
                       },
                  4: {
                       -3: -s(7/120),
                       -2: -s(49/480),
                       -1: -s(21/160),
                       0: -s(7/48),
                       1: -s(7/48),
                       2: -s(21/160),
                       3: -s(49/480),
                       4: -s(7/120),
                       },
                   3: {
                       -2: s(1/288),
                       -1: s(1/96),
                       0: s(1/48),
                       1: s(5/144),
                       2: s(5/96),
                       3: s(7/96),
                       4: s(7/72),
                       },
                   },
               },
           3: {
               'sigma_plus': {
                   4: {
                       -3: s(5/672),
                       -2: s(5/224),
                       -1: s(5/112),
                       0: s(25/336),
                       1: s(25/224),
                       2: s(5/32),
                       3: s(5/24),
                       },
                  3: {
                       -3: s(3/32),
                       -2: s(5/32),
                       -1: s(3/16),
                       0: s(3/16),
                       1: s(5/32),
                       2: s(3/32),
                       },
                   2: {
                       -3: s(5/14),
                       -2: s(5/21),
                       -1: s(1/7),
                       0: s(1/14),
                       1: s(1/42),
                       },
                   },
               'pi': {
                   4: {
                       -3: -s(5/96),
                       -2: -s(5/56),
                       -1: -s(25/224),
                       0: -s(5/42),
                       1: -s(25/224),
                       2: -s(5/56),
                       3: -s(5/96),
                       },
                  3: {
                       -3: -s(9/32),
                       -2: -s(1/8),
                       -1: -s(1/32),
                       0: s(0),
                       1: s(1/32),
                       2: s(1/8),
                       3: s(9/32),
                       },
                   2: {
                       -2: s(5/42),
                       -1: s(4/21),
                       0: s(3/14),
                       1: s(4/21),
                       2: s(5/42),
                       },
                   },
               'sigma_minus': {
                   4: {
                       -3: s(5/24),
                       -2: s(5/32),
                       -1: s(25/224),
                       0: s(25/336),
                       1: s(5/112),
                       2: s(5/224),
                       3: s(5/672),
                       },
                  3: {
                       -2: -s(3/32),
                       -1: -s(5/32),
                       0: -s(3/16),
                       1: -s(3/16),
                       2: -s(5/32),
                       3: -s(3/32),
                       },
                   2: {
                       -1: s(1/42),
                       0: s(1/14),
                       1: s(1/7),
                       2: s(5/21),
                       3: s(5/14),
                       },
                   },
               },
           },
       'D1': {
           4: {
               'sigma_plus': {
                  4: {
                       -4: s(1/12),
                       -3: s(7/48),
                       -2: s(3/16),
                       -1: s(5/24),
                       0: s(5/24),
                       1: s(3/16),
                       2: s(7/48),
                       3: s(1/12),
                       },
                   3: {
                       -4: s(7/12),
                       -3: s(7/16),
                       -2: s(5/16),
                       -1: s(5/24),
                       0: s(1/8),
                       1: s(1/16),
                       2: s(1/48),
                       },
                   },
               'pi': {
                  4: {
                       -4: -s(1/3),
                       -3: -s(3/16),
                       -2: -s(1/12),
                       -1: -s(1/48),
                       0: s(0),
                       1: s(1/48),
                       2: s(1/12),
                       3: s(3/16),
                       4: s(1/3),
                       },
                   3: {
                       -3: s(7/48),
                       -2: s(1/4),
                       -1: s(5/16),
                       0: s(1/3),
                       1: s(5/16),
                       2: s(1/4),
                       3: s(7/48),
                       },
                   },
               'sigma_minus': {
                  4: {
                       -3: -s(1/12),
                       -2: -s(7/48),
                       -1: -s(3/16),
                       0: -s(5/24),
                       1: -s(5/24),
                       2: -s(3/16),
                       3: -s(7/48),
                       4: -s(1/12),
                       },
                   3: {
                       -2: s(1/48),
                       -1: s(1/16),
                       0: s(1/8),
                       1: s(5/24),
                       2: s(5/16),
                       3: s(7/16),
                       4: s(7/12),
                       },
                   },
               },
           3: {
               'sigma_plus': {
                   4: {
                       -3: -s(1/48),
                       -2: -s(1/16),
                       -1: -s(1/8),
                       0: -s(5/24),
                       1: -s(5/16),
                       2: -s(7/16),
                       3: -s(7/12),
                       },
                  3: {
                       -3: -s(1/16),
                       -2: -s(5/48),
                       -1: -s(1/8),
                       0: -s(1/8),
                       1: -s(5/48),
                       2: -s(1/16),
                       },
                   },
               'pi': {
                   4: {
                       -3: s(7/48),
                       -2: s(1/4),
                       -1: s(5/16),
                       0: s(1/3),
                       1: s(5/16),
                       2: s(1/4),
                       3: s(7/48),
                       },
                  3: {
                       -3: s(3/16),
                       -2: s(1/12),
                       -1: s(1/48),
                       0: s(0),
                       1: -s(1/48),
                       2: -s(1/12),
                       3: -s(3/16),
                       },
                   },
               'sigma_minus': {
                   4: {
                       -3: -s(7/12),
                       -2: -s(7/16),
                       -1: -s(5/16),
                       0: -s(5/24),
                       1: -s(1/8),
                       2: -s(1/16),
                       3: -s(1/48),
                       },
                  3: {
                       -2: s(1/16),
                       -1: s(5/48),
                       0: s(1/8),
                       1: s(1/8),
                       2: s(5/48),
                       3: s(1/16),
                       },
                   },
               },
           }
       }

def checkEntriesDME(Fprime_dict):
    
    cumsum = 0
    
    for mF_dict in Fprime_dict.values():
        for dme in mF_dict.values():
            cumsum += dme**2
            
    return cumsum
            
def saturation_intensity(line, polarization):
    if (line == 'cycling') and (polarization == 'isotropic'):
        return quantity(2.7059*10, 'W/m2')
    elif (line == 'cycling') and (polarization == 'circular'):
        return quantity(1.1023*10, 'W/m2')
    elif (line == 'D2') and (polarization == 'linear'):
        return quantity(1.6536*10, 'W/m2')
    elif (line == 'D1') and (polarization == 'linear'):
        return quantity(2.4981*10, 'W/m2')
    else:
        raise Exception('Saturation intensity not precalcualted for this combination of frequency, line, and polarization.')

physical_properties = {
    'atomic_number': (55, ''),
    'total_nucleons': (133, ''),
    'relative_natural_abundance': (1, ''),
    'nuclear lifetime': (np.inf, 's'),
    'atomic_mass': (2.20694695e-25, 'kg'),
    'density_stp': (1.93 * 1e6 * 1e-3, 'kg/m3'),
    'melting_point': (28.44 + 273.15, 'K'),
    'boiling_point': (671 + 273.15, 'K'),
    'specific_heat_capacity': (0.242 * 1e3, 'J/kg*K'),
    'molar_heat_capacity': (32.210, 'J/mol*K'),
    'vapor_pressure_stp': (1.3e-6, 'torr'),
    'nuclear_spin': (7/2, ''),
    'ionization_limit': (3.89390532 * 1.602e-19, 'J'),
    }

d2_properties = {
    'frequency': (351.72571850 * 1e12, 'Hz'),
    'transition_energy': (1.454620542 * 1.602e-19, 'J'),
    'wavelength': (852.34727582 * 1e-9, 'm'),
    'lifetime': (30.473 * 1e-9, 's,'),
    'linewidth': (2 * np.pi * 5.2227 * 1e6, 'Hz'),
    'oscillator_strength': (0.7148, ''),
    'recoil_velocity': (3.5225 * 1e-3, 'm/s'),
    'recoil_energy': (2*np.pi * 2.0663 * 1e3, 'Hz'),
    'recoil_temperature': (198.34 * 1e-9, 'K'),
    'doppler_shift': (2*np.pi * 4.1327 * 1e3, 'Hz'),
    'doppler_temperature': (125 * 1e-6, 'K'),
    'Jg': (0.5, ''),
    'Je': (1.5, ''),
    'dme': (3.8014e-29, 'C*m'),
    }

d1_properties = {
    'frequency': (335.116048807 * 1e12, 'Hz'),
    'transition_energy': (1.385928475 * 1.602e-19, 'J'),
    'wavelength': (894.59295986 * 1e-9, 'm'),
    'lifetime': (34.894 * 1e-9, 's'),
    'linewidth': (2*np.pi * 4.5612 * 1e6, 'Hz'),
    'oscillator_strength': (0.3438, ''),
    'recoil_velocity': (3.3561 * 1e-3, 'm/s'),
    'recoil_energy': (2*np.pi * 1.8758 * 1e3 , 'Hz'),
    'recoil_temperature': (180.05 * 1e-9, 'K'),
    'doppler_shift': (2*np.pi * 3.7516 * 1e3, 'Hz'),
    'Jg': (0.5, ''),
    'Je': (0.5, ''),
    'dme': (2.7020e-29, 'C*m'),
    }

hyperfine_constants = {
    'magnetic_dipole_6p12': ()}

transition_properties = {
    'D1': d1_properties,
    'D2': d2_properties,
    }

cesium = Atom(physical_properties, transition_properties)
cesium.saturation_intensity = saturation_intensity
cesium.dme_coeffs = dme_coeffs

if __name__ == '__main__':
    # check all dmes line / F / polarization / Fprime / mF
    for line, F_dict in dme_coeffs.items():
        for F, polarization_dict in F_dict.items():
            for polarization, Fprime_dict in polarization_dict.items():
                ans = checkEntriesDME(Fprime_dict)
                print(line, F, polarization, ans)