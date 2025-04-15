# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:15:33 2021

@author: Shankari
"""

from arc import *
from arc.calculations_atom_single import DynamicPolarizability as dPol
from analysis.calculation.dipoletrap import trapDepth
import time


c = 2.99792458e8 #m/s
eps0 = 8.85418781762039e-12 # A^2 s^4 kg^-1 m^-3
h = 6.62607015e-34 # J Hz^-1
a0 = 5.29177210903e-11  #m
me = 9.1093837015e-31 #kg
e = 1.602176634e-19
hartreeFrequency = (h/(2*np.pi))/(me * (a0)**2)
hartreeEnergy = hartreeFrequency * (h/(2*np.pi))
polAU = e**2 * a0**2 / hartreeEnergy

beamPower = 5 #W
beamWaist = 100 #in um

# beamIntensity = beamPower/(np.pi * (beamWaist)**2) #W/um^2
# beamIntensityWmm2 = beamIntensity * (1e3)**2
# beamIntensitySI = beamIntensity * (1e6)**2 # in W/m^2

beamIntensity = 1 #mW/cm^2
beamIntensitySI = beamIntensity * (100**2) * (1000)**-1

atom = Cesium()
wavelength = [780e-9] #np.arange(400e-9, 1200.001e-9, 5e-11)

statesOfInterest = ['gs','e2','e1']
calculations = ['polMethod',
                'scalar polarizability', 'vector polarizability', 'tensor polarizability', 
                'core polarizability', 'ponderomotive polarizability', 'closest state']
calcData = {key:{} for key in calculations}
quantumNumbers = {key:{} for key in statesOfInterest}

# stuff for 6S_1/2
quantumNumbers['gs']['n'] = atom.groundStateN
quantumNumbers['gs']['l'] = 0
quantumNumbers['gs']['j'] = 0.5

# stuff for 6P_1/2
quantumNumbers['e1']['n'] = atom.groundStateN
quantumNumbers['e1']['l'] = 1
quantumNumbers['e1']['j'] = 0.5

# stuff for 6P_3/2
quantumNumbers['e2']['n'] = atom.groundStateN
quantumNumbers['e2']['l'] = 1
quantumNumbers['e2']['j'] = 1.5

allScalarPolGS = []
allScalarPolE2 = []

start = time.time()

for wl in wavelength:
    if int(np.mod(np.round(wl/5e-11), 1000)) == 0:
        print('current wavelength = ', wl*1e9, ' nm')
    for state in statesOfInterest:
        calcData['polMethod'][state] = dPol(atom, quantumNumbers[state]['n'], 
                                                      quantumNumbers[state]['l'], 
                                                      quantumNumbers[state]['j'])
        
        calcData['polMethod'][state].defineBasis(atom.groundStateN, atom.groundStateN+40)
    
        calcData['scalar polarizability'][state],     \
            calcData['vector polarizability'][state], \
            calcData['tensor polarizability'][state], \
            calcData['core polarizability'][state],   \
            calcData['ponderomotive polarizability'][state], \
            calcData['closest state'][state]  = calcData['polMethod'][state].getPolarizability(wl, units = 'SI')
        
        if state == 'gs':
            allScalarPolGS.append(calcData['scalar polarizability'][state])
        elif state == 'e2':
            allScalarPolE2.append(calcData['scalar polarizability'][state])

end = time.time()
trapDepthJacob = trapDepth(intensity = beamIntensitySI, method='steck_quantum', wavelength = wavelength[0])
trapDepthARC = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['gs']) * beamIntensitySI

polarizabilitySteck = 0.1001 / (100**2) * ((c/(895e-9**2))/((c/((895e-9)**2))-(c/((wavelength[0])**2))))
trapDepthSteck = 1/(2*eps0*c) * polarizabilitySteck * beamIntensitySI

# lightShiftD2Steck = 0.3086 / (100**2) * ((c/(895e-9**2))/((c/((895e-9)**2))-(c/((1064e-9)**2))))
# lightShiftD21064ARC = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['e2']-calcData['scalar polarizability']['gs']) * beamIntensitySI

print('trapDepth Jacob:', trapDepthJacob/1e6, ' MHz')
print('trapDepth ARC  :', trapDepthARC/1e6, ' MHz')
print('trapDepth Steck  :', trapDepthSteck/1e6, ' MHz')

# print('lightShift D2 Steck:', lightShiftD2Steck/1e6, 'MHz')
# print('lightShift D2 ARC:', lightShiftD21064ARC/1e6, 'MHz')

wavelengths = np.multiply(wavelength, 1e9)
trapDepthAllGSARC = np.divide(1/(2*eps0*c) * np.real(allScalarPolGS) * beamIntensitySI, 1e6)
trapDepthAllE2ARC = np.divide(1/(2*eps0*c) * np.real(allScalarPolE2) * beamIntensitySI, 1e6)


# fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, ncols = 1, figsize = (20, 20))
# plt.subplots_adjust(hspace=0.25)

# ax1.scatter(wavelengths, trapDepthAllGSARC)
# ax1.set_xlabel('wavelength (nm)', fontsize = 16)
# ax1.set_ylabel('trap depth (MHz)', fontsize = 16)
# ax1.set_xlim([min(wavelengths), max(wavelengths)])
# ax1.set_ylim([-15, 15])
# ax1.tick_params(labelsize = 16)
# ax1.set_title('6$^2$S$_{1/2}$ trap depth at 5 W, 50 um waist', fontsize = 20)
# ax1.grid()

# ax2.scatter(wavelengths, trapDepthAllE2ARC)
# ax2.set_xlabel('wavelength (nm)', fontsize = 16)
# ax2.set_ylabel('trap depth (MHz)', fontsize = 16)
# ax2.set_xlim([min(wavelengths), max(wavelengths)])
# ax2.set_ylim([-15, 15])
# ax2.tick_params(labelsize = 16)
# ax2.set_title('6$^2$P$_{3/2}$ trap depth at 5 W, 50 um waist', fontsize = 20)
# ax2.grid()

# ax3.scatter(wavelengths, (trapDepthAllE2ARC-trapDepthAllGSARC))
# ax3.set_xlabel('wavelength (nm)', fontsize = 16)
# ax3.set_ylabel('light shift (MHz)', fontsize = 16)
# ax3.set_xlim([min(wavelengths), max(wavelengths)])
# ax3.set_ylim([-15, 15])
# ax3.tick_params(labelsize = 16)
# ax3.set_title('6$^2$P$_{3/2}$ - 6$^2$S$_{1/2}$ lightshift at 5 W, 50 um waist', fontsize = 20)
# ax3.grid()


# print('elapsed time: ', end-start)

# fig.savefig(r'Z:\Experiments\rydberglab\Squeezing_dds_mw\2021\11\08\lightShiftCalc\lightShiftARCTestCalc.png', dpi=300)
# np.save(r'Z:\Experiments\rydberglab\Squeezing_dds_mw\2021\11\08\lightShiftCalc\groundStateDepths.npy', trapDepthAllGSARC)
# np.save(r'Z:\Experiments\rydberglab\Squeezing_dds_mw\2021\11\08\lightShiftCalc\excitedStateDepths.npy', trapDepthAllE2ARC)
# np.save(r'Z:\Experiments\rydberglab\Squeezing_dds_mw\2021\11\08\lightShiftCalc\wavelengths.npy', wavelengths)

differentialShift = -2.26e-2*2 #Hz cm^2 W^-1- from PRA 79, 013404 (2009) (theory) and PRA 57, 436 (1998)
differentialShiftSI = differentialShift * (100)**-2


powers = np.linspace(0, 5, 100)
ourPower = 3.75/2
intensities = np.multiply(2, np.divide(powers, np.pi * (beamWaist)**2)) #W/um**2 average intensity over lattice
intensitiesSI = intensities * (1e6)**2
ourIntensitySI = np.multiply(2 * (1e6)**2, np.divide(ourPower, np.pi * (beamWaist)**2))

trapDepthsGS780ARC = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['gs']) * intensitiesSI
trapDepthsE2780ARC = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['e2']) * intensitiesSI
totalStarkShiftD2 = trapDepthsGS780ARC-trapDepthsE2780ARC
differentialStarkShiftGS = 1/(2) * differentialShiftSI * intensitiesSI

trapDepthsGS780_ourPower = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['gs']) * ourIntensitySI
trapDepthsE2780_ourPower = 1/(2*eps0*c) * np.real(calcData['scalar polarizability']['e2']) * ourIntensitySI
totalStarkShiftD2_ourPower = trapDepthsGS780_ourPower-trapDepthsE2780_ourPower
differentialStarkShiftGS_ourPower = 1/(2) * differentialShiftSI * ourIntensitySI

fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows = 1, ncols = 4, figsize = (25, 5))
fs = 11
ax1.plot(powers, totalStarkShiftD2/1e6)
ax1.scatter(ourPower, totalStarkShiftD2_ourPower/1e6, color = 'r', marker = 's', s=24, label = f'expected lightshift: {totalStarkShiftD2_ourPower/1e6:0.3f} MHz')
ax1.set_xlabel('total power (W)')
ax1.set_ylabel('total D2 stark shift (MHz)')
ax1.grid()
ax1.legend(fontsize = fs)
ax2.plot(powers, trapDepthsGS780ARC/1e6)
ax2.scatter(ourPower, trapDepthsGS780_ourPower/1e6, color = 'r', marker = 's', s=24, label = f'expected gs trap depth: {trapDepthsGS780_ourPower/1e6:0.3f} MHz')
# ax3.scatter(trapDepthsGS780_ourPower/1e6, color = 'r', marker = 's', s=24, label = f'expected gs trap depth: {trapDepthsGS780_ourPower/1e6:0.3f} MHz')
ax2.set_xlabel('total power (W)')
ax2.set_ylabel('GS trap depth (MHz)')
ax2.grid()
ax2.legend(fontsize = fs)
ax3.plot(trapDepthsGS780ARC/1e6, differentialStarkShiftGS)
ax3.scatter(trapDepthsGS780_ourPower/1e6, differentialStarkShiftGS_ourPower, color = 'r', marker = 's', s=24, label = f'differential shift slope: {differentialStarkShiftGS_ourPower/trapDepthsGS780_ourPower*1e4:0.3f} *1e-4')
ax3.set_xlabel('GS trap depth (MHz)')
ax3.set_ylabel('differential GS stark shift (Hz)')
ax3.grid()
ax3.legend(fontsize = fs)
ax4.plot(trapDepthsGS780ARC/1e6, totalStarkShiftD2/1e6)
ax4.scatter(trapDepthsGS780_ourPower/1e6, totalStarkShiftD2_ourPower/1e6, color = 'r', marker = 's', s=24, label = f'slope: {totalStarkShiftD2_ourPower/trapDepthsGS780_ourPower:0.3f}')
ax4.set_xlabel('GS trap depth (MHz)')
ax4.set_ylabel('total D2 stark shift (MHz)')
ax4.grid()
ax4.legend(fontsize = fs)