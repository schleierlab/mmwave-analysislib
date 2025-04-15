# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 18:45:57 2017

@author: Stanford University
"""


import numpy as np
from numpy import exp, sin, cos, pi, sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def line(x, m, b):
    return m*x + b

def line_pi2(x, m, x0):
    return m*(x-x0) + 0.5

def rabi_osc(t, A, t0, f, tau, offset, offset0):
    return A * exp(-(t-t0)/tau)* ((sin(2*pi * f * (t-t0)/2))**2 + offset0) + offset

def rabi_osc_noDecayOffset(t, A, t0, f, tau, offset):
    return A * exp(-(t-t0)/tau)* ((sin(2*pi*f*(t-t0)/2))**2) + offset

def rabi_osc_no_decay(t, A, t0, f):
    return A * (sin(2*pi*f*(t-t0)/2)**2)

def decaying_sine(t, A,t0, f, tau, offset):
    return A * exp(-(t-t0)/tau)* ((sin(2*pi*f*(t-t0)/2))**2) + offset

# def rabi_osc_no_decay(t, A, t0, f, offset):
#     return A * (sin(2*pi*f*(t-t0)/2))**2 + offset
    
def ramsey_fringes(t, A, t0, f, tau, offset):
    return A * exp(-(t-t0)/tau)*(cos(2*pi*f*(t-t0))) + offset

def spin_echo(phi, A, phi0, offset):
    return A * sin(pi*phi/360 + phi0)**2 + offset

def sin_fit(phi, phi0, A, offset):
    return A/2 * sin(pi*(phi-phi0)/180) + offset

def cos_fit(phi, phi0, A, offset):
    return A/2 * cos(pi*(phi-phi0)/180) + offset

def sin_fit_2(phi, phi0, phi1, A, A2, omega, offset):
    return A/2 * sin(pi*(phi-phi0)/180) + A2/2 * sin(omega*pi*(phi-phi1)/180) + offset

def sin_fit_freq(t, omega, t0, A, offset):
    return A/2 * sin(omega*(t-t0)) + offset

def quadratic(x,a,b,c):
    return a*(x-b)**2+c

def twoD_Quadratic(xy, x_amp, y_amp, xo, yo, offset):
    (x, y) = xy
    xo = float(xo)                                                              
    yo = float(yo)
    q = offset + x_amp*(x - xo)**2 + y_amp*(y - yo)**2 
    return q.ravel()


def quadlin(x,a,m,b):
    return a*(x)**2 + m*x + b

def ramsey_coherence(t, coherence_time, detuning, phase, offset):
    return np.exp(-t**2/(2*coherence_time**2)) * 0.5*cos(2*pi*detuning*t + phase) + offset

def one_state_fidelity(n, f, a0, offset):
    
    b0 = 1 - a0
    
    update_matrix = np.array([
        [1-f, f],
        [f, 1-f]],
        )
    
    def propagate(n, a, b):
        recursion_matrix = np.linalg.matrix_power(update_matrix, int(n))
        state = np.array([a, b])
        return np.dot(recursion_matrix, state)
        
    if hasattr(n, '__len__'):
        state = np.zeros((2, len(n)))

        for idx, val in enumerate(n):
            state[:, idx] = propagate(val, a0, b0)
            
        a = state[0, :]
        b = state[1, :]
    else:
        a, b = propagate(n, a0, b0)
    
    f4 = a + offset
    
    return f4


def two_state_fidelity(n, f12, f23, a0, b0, offset):
    # print(n)
    c0 = 1 - a0 - b0
    
    sweep_right = np.array([
        [1-f12, f12, 0],
        [f12*(1-f23), (1-f12)*(1-f23), f23],
        [f12*f23, (1-f12)*f23, 1-f23]],
        )
    
    sweep_left = sweep_right.T
    
    def propagate(n, a, b, c):
        round_trip = np.dot(sweep_left, sweep_right)
        recursion_matrix = np.linalg.matrix_power(round_trip, int(n//2))
        
        if 1 <= n%2 <2:
            recursion_matrix = np.dot(sweep_right, recursion_matrix)
        
        state = np.array([a, b, c])
        return np.dot(recursion_matrix, state)
        
    if hasattr(n, '__len__'):
        state = np.zeros((3, len(n)))

        for idx, val in enumerate(n):
            state[:, idx] = propagate(val, a0, b0, c0)
            
        a = state[0, :]
        b = state[1, :]
        c = state[2, :]
        
    else:
        a, b, c = propagate(n, a0, b0, c0)
    
    f4 = a + c + offset
    
    return f4


def rabi_spec(d, t, Frabi, C, K, f0):
    #d = f-f_0#detuning
    #t= 40  #length of the Rabi pulse
    return K*(Frabi**2/(Frabi**2+(d-f0)**2))*(sin(pi*sqrt(Frabi**2+(d-f0)**2)*t))**2+C
    
def tof_temp(t, T, x0):
    #t is in s
    #x0 in meters
    kb=1.38*10^-23
    m=87*1.66*10^-27
    return sqrt(x0 + (T*kb/m)*t**2)
    
def exp_decay(t,A, tau, offset):
    t0=0
    return A*exp(-(t-t0)/tau) + offset

def exp_decay_noOffset(t,A, tau):
    return A*exp(-(t)/tau)

def gaussian(x, x0, A, sigma):    
    return A*exp( -((x-x0)**2)/(2*sigma**2) )

def gaussianOffset(x, x0, A, sigma,B):    
    # note that the gaussian component here integrates to sqrt(2pi) * sigma
    return A*exp(-((x-x0)**2)/(2*sigma**2)) + B

def gaussianOffsetx5(x, x0_1, A1, sigma1, x0_2, A2, sigma2, x0_3, A3, sigma3, x0_4, A4, sigma4, x0_5, A5, sigma5,B):    
    return A1*exp( -((x-x0_1)**2)/(2*sigma1**2) ) + A1*exp( -((x-x0_1)**2)/(2*sigma1**2) ) + A1*exp( -((x-x0_1)**2)/(2*sigma1**2) ) + A1*exp( -((x-x0_1)**2)/(2*sigma1**2) ) + A1*exp( -((x-x0_1)**2)/(2*sigma1**2) ) + B
    
def lorentzian(x,A, full_width, x0):
    return A/(1+(2*(x-x0)/full_width)**2)

def lorentzian_phase(x, A, full_width, x0):
    return A*(x-x0)/(1+(2*(x-x0)/full_width)**2)

def lorentzian_phase_2(x, omega, full_width, x0):
    return -(omega**2)*(x-x0)/(full_width**2+4*(x-x0)**2)

def lorentzian_phase_3(x, omega, x0):
    return omega**2*(x-x0)/(1**2+4*(x-x0)**2)

def lorentzian_offset(x, A, full_width, x0, offset):
    return A/(1+(2*(x-x0)/full_width)**2)+offset

def gaussian_offset(x, A, sigma, x0, offset):
    return A*exp( -((x-x0)**2)/(2*sigma**2) ) + offset

def double_lorentzian_offset(x, A, full_width_1, x0_1,B, full_width_2, x0_2,offset):
    return A/(1+(2*(x-x0_1)/full_width_1)**2)+B/(1+(2*(x-x0_2)/full_width_2)**2)+offset

def quadraticStarkShift(x, alpha, E_0, f_0):
    return 0.5*alpha*(x-E_0)**2+f_0

def dressedPotential(x, x0, Omega):
    return Omega**4/(8*(x-x0)**3)

def dressedPotential_1(x, x0, K):
    #Dressed potential including the number of atoms in the critical radius
    #where K=(C6)**(1/2)*rho*Omega**4,
    #where C6 is an averaged C6 coefficient and
    #rho is the atom number density.
    return K/(8*(x-x0)**(3.5))

def dressedPotential_2(x, K):
    #Dressed potential including the number of atoms in the critical radius
    #where K=(C6)**(1/2)*rho*Omega**4,
    #where C6 is an averaged C6 coefficient and
    #rho is the atom number density.
    return K/(8*(x)**(3.5))

def lightShift(x,x0,Omega):
    return Omega**2/(4*(x-x0))


def lightShiftNoOffset(det, A):
    return A**2 / (4*det)

def oneAndTwoBodyLoss(t,singleBodyLoss,twoBodyLoss,initialNumAtoms):
    A = singleBodyLoss 
    B = twoBodyLoss
    C = initialNumAtoms/(A+B*initialNumAtoms)
    # number of atoms is n(t)
    # with those variables, this function should model n'(t) = -A*n -B*n^2 => n(t) = A*C/(Exp(At)-B*C) for a constant C
    return A*C/(exp(A*t)-B*C)

def oneD_Gaussian_withRotation(xy, amplitude, xo, yo, sigma_x, sigma_y, offset, theta):
    (x, y) = xy
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)    
    g = offset + amplitude*exp( - a*((x-xo)**2))                          
    return g.ravel()


def twoD_Gaussian_withRotation(xy, amplitude, xo, yo, sigma_x, sigma_y, offset, theta):
    (x, y) = xy
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)   
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)    
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))                                   
    return g.ravel()


def twoD_Gaussian_noRotation(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    thetaVal=0
    (x, y) = xy
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (cos(thetaVal)**2)/(2*sigma_x**2) + (sin(thetaVal)**2)/(2*sigma_y**2)   
    b = -(sin(2*thetaVal))/(4*sigma_x**2) + (sin(2*thetaVal))/(4*sigma_y**2)    
    c = (sin(thetaVal)**2)/(2*sigma_x**2) + (cos(thetaVal)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))                                   
    return g.ravel()


def Bcomp_fit(Ix, I0x, B0perp, beta):
    #I0x is the current that zeroes the field along x
    #B0perp is the residual field in the yz plane
    #beta is the conversion from current to magnetic field (G/A)
    dmu = 7*0.35 #MHz/G, for fitting
    return dmu*sqrt((beta*(Ix-I0x))**2+B0perp**2)

def cloudRadiusCesium(time, temp, sigma_0):
    '''
    Cloud radius of a cloud of cesium atoms versus time and the initial size of the cloud.
    '''
    #temp is the atom temperature
    #sigma_0 is initial cloud size
    kB = 1.38*10**-23
    m = 2.206e-25 #in kg Cesium!
    return sqrt(sigma_0**2 + kB*temp/m*time**2)

def cloudRadiusCesiumOffset(time,temp,sigma_0, t_0):
    #temp is the atom temperature
    #sigma_0 is initial cloud size
    kB = 1.38*10**-23
    m = 2.206e-25 #in kg Cesium!
    return sqrt(sigma_0**2 + kB*temp/m*(time-t_0)**2)

def binomVarP(p, N, a, b):
    return 4*p*(1-p)/N + a*p**2 + b

def binomVarP_withoutAtomNoise(p, N, b):
    return 4*p*(1-p)/N + b

def varFraction(f4, N, pulseNoise, atomNoise, c):
    '''
    Similar to binomVarP this function returns Var(f4-f3) vs. f4, but with
    more explicitly defined noise sources.

    Parameters
    ----------
    f4 : float
        Fraction of atoms in F=4.
    N : float
        Number of atoms.
    pulseNoise : float
        Fractional fluctuations in the pulse angle.
    atomNoise : float
        Fractional fluctuations in F=4 atoms ending up in the F=3 image.

    Returns
    -------
    float
        Variance.

    '''
    
    binomVar = 4*f4*(1-f4)/N
    pulseVar = 16*f4*(1-f4)*np.arcsin(np.sqrt(f4))**2
    atomVar = 4*f4**2
    
    return binomVar + pulseNoise*pulseVar+atomVar*atomNoise+c
    

def binomVarN(N,a,b,c):
    #Variance versus Atom Number
    return a/N**2+b/N+c

def sin_noise(theta, A, theta0, offset):
    return A*sin(theta - theta0)**2 + offset

def sin_squeezing(theta, A, projectionNoise):
    B = np.sqrt(1+A**2)
#    theta0 = np.arcsin((1-B)/A)
#    theta0 = np.nan_to_num(theta0)
    theta0 = -pi/2
    return projectionNoise*(A*sin(2*pi*theta + theta0) + B)

def gaussianBeamRadius(z, w0, z0):
    # wavelength = 0.852e-6#1.064e-6
    wavelength = 780e-9
    return w0 * np.sqrt(1 + wavelength**2 *(z-z0)**2/(pi**2*w0**4))

######## for fitting long vs short pulses- coherent vs incoherent model


def Qratio(T, gammaL, tgap, gamma):
    return (gammaL/(2*T*(gamma**2))) * ((gamma*T) - (1-np.exp(-gamma*T)) - (0.5 * (np.exp(-gamma*tgap)) * ((1-np.exp(-gamma*T))**2)))


def chiTot(T, gammaL, tgap, gamma):
    return 1+Qratio(T, gammaL, tgap, gamma)


def Qtot(T, gammaL, tgap, gamma):
    return 2*T*chiTot(T, gammaL, tgap, gamma)


def QtotN(T, gammaL, tgap, n, gamma):
    return np.multiply(Qtot(np.divide(T,n), gammaL, tgap, gamma),n)


def shortFit(T, a, gammaL, tgap, numSpinEcho, gamma, tauR):
    return a * QtotN(T, gammaL, tgap, numSpinEcho, gamma) / tauR

def longFit(T, a, gammaL, tgap, numSpinEcho, gamma, tauR):
    return a * QtotN(T, gammaL, tgap, numSpinEcho, gamma) / tauR

def incoherentExcitation(allData, a, gammaL, tgap_long, tgap_short, numSpinEcho_long, numSpinEcho_short, gamma, tauR, timesLong,):
    extract1 = allData[:timesLong]
    extract2 = allData[timesLong:]
    
    
    result1 = longFit(extract1, a, gammaL, tgap_long, numSpinEcho_long, gamma, tauR)
    result2 = shortFit(extract2, a, gammaL, tgap_short, numSpinEcho_short, gamma, tauR)
    
    return np.append(result1, result2)


###### for fitting two datasets with lines with different slopes, but constrained offset ###########
def fitTwoLinesWithConstrainedOffset(allX, m1, m2, b, len_density1data):
    extract1 = allX[:len_density1data]
    extract2 = allX[len_density1data:]
    
    result1 = line(extract1, m1, b)
    result2 = line(extract2, m2, b)
    
    return(np.append(result1, result2))

def fitLightshiftSaturation(f4, fsat, Umax):
    return Umax/2 * (1+np.exp(np.divide(-1*f4, fsat)))

def fitLightshiftSaturationConstrainedOffset(f4, fsat1, fsat2, Umax, len_density1data):
    extract1 = f4[:len_density1data]
    extract2 = f4[len_density1data:]
    
    result1 = fitLightshiftSaturation(extract1, fsat1, Umax)
    result2 = fitLightshiftSaturation(extract2, fsat2, Umax)
    
    return(np.append(result1, result2))

def integrated_gaussian_beam(x, center, amplitude, waist, offset):
    return amplitude * np.exp((-2*(x-center)**2)/(waist**2)) + offset

def lattice_contrast_error(x, center, waist, amplitude, contrast, spacing, phase, offset):
    oscillation = cos(2*pi * (x-center)/spacing + phase)
    intensity = amplitude * np.exp((-2*(x-center)**2)/(waist**2))
    
    return intensity * (1 + contrast*oscillation) + offset

def lattice_power_imbalance(x, center, waist, amplitude, fraction, spacing, phase, offset):
    power_term = np.sqrt(fraction * (1-fraction))
    contrast = 2 * power_term
    
    oscillation = cos(2*pi * (x-center)/spacing + phase)
    intensity = amplitude * np.exp((-2*(x-center)**2)/(waist**2))
    
    return intensity * 0.5*(1 + contrast*oscillation) + offset

def lattice_polarization_error(x, center, waist, amplitude, alpha, spacing, phase, offset):
    wavelength = 0.780
    beta = 0
    polarization_term = cos(alpha)*cos(beta) + sin(alpha)*sin(beta)*(2*(wavelength/(2*spacing))**2 - 1)
    contrast = 2 * polarization_term
    
    oscillation = cos(2*pi * (x-center)/spacing + phase)
    intensity = amplitude * np.exp((-2*(x-center)**2)/(waist**2))
    
    return intensity * (1 + contrast*oscillation) + offset

def lattice_generalized_error(x, center, waist, amplitude, fraction, alpha, spacing, phase, offset):
    wavelength = 0.780
    beta = 0

    power_term = np.sqrt(fraction * (1-fraction)) 
    polarization_term = cos(alpha)*cos(beta) + sin(alpha)*sin(beta)*(2*(wavelength/(2*spacing))**2 - 1)
    contrast =  2 * power_term * polarization_term
    
    oscillation = cos(2*pi * (x-center)/spacing + phase)
    intensity = amplitude * np.exp((-2*(x-center)**2)/(waist**2))
    
    return intensity * (1 + contrast*oscillation) + offset

def lattice_offset_beams(x, center1, center2, amplitude1, amplitude2, waist1, waist2, spacing, phase, offset):
    I1 = integrated_gaussian_beam(x, center1, amplitude1, waist1, 0)
    I2 = integrated_gaussian_beam(x, center2, amplitude2, waist2, 0)
    
    intensity = I1 + I2
    contrast = 2*np.sqrt(I1*I2)/(I1+I2)    
    oscillation = cos(2*pi * x/spacing + phase)
    
    return intensity * (1 + contrast*oscillation) + offset

def trapezoidalPulse(x, x1, x2, x3, m1, m2, y1):
    initFlat = lambda x: y1
    firstStep = lambda x: m1*(x-x1) + y1
    nextFlat = lambda x: m1*(x2-x1)+y1
    secondStep = lambda x: m2*(x-x3) + m1*(x2-x1)+y1
    
    x4 = ((x1-x2) * (m1/m2)) + x3 # guarantee returns to same as initlevel
    lastFlat = lambda x: m2*(x4-x3) + m1*(x2-x1)+y1
    
    y =  np.piecewise(x, [x < x1, 
                          ((x >= x1) & (x < x2)), 
                          ((x >= x2) & (x < x3)), 
                          ((x >= x3) & (x < x4)), 
                          x >= x4,
                          ], 
                         [initFlat, 
                          firstStep, 
                          nextFlat,
                          secondStep,
                          lastFlat])
    return y


if __name__ == "__main__":
#    A = 2
#    theta = np.linspace(-0.5,1,1000)
#    y = sin_squeezing(theta, A, 1)
#    
#    fig, ax = plt.subplots(figsize=(9,6))
#    plt.plot(theta/pi,y)
#    plt.grid()
#
#    plt.xlabel(r'$\theta$ (rad)')
#    plt.ylabel('noise / projection noise')
#    ax.set_yscale('log')
#    ax.set_ylim(bottom=1e-2, top=1e2)
    
    pixelSize = 5.5;
    magnification = 750./175;
    ysig = np.array([14, 7, 5, 5, 10, 15,11,5]);
    waist = ysig*2*pixelSize/magnification*1e-6;
    objectiveHeight = np.array([4.845, 4.45, 4.2, 3.6, 3.3, 3.0,4.65,3.9])*1e-3;
    
    coeffs, covar = curve_fit(gaussianBeamRadius, objectiveHeight,waist, p0=[10e-6,3.8e-3])
    
    xvals = np.linspace(np.min(objectiveHeight),np.max(objectiveHeight),100)
    fitlabel = r'w0 = {:.2f} um, z0 = {:.2f} mm'
    
    plt.scatter(objectiveHeight*1e3,waist*1e6)
    plt.plot(xvals*1e3,gaussianBeamRadius(xvals,*coeffs)*1e6,label = fitlabel.format(coeffs[0]*1e6,coeffs[1]*1e3))
    plt.ylim(bottom=0)
    plt.legend()
    plt.xlabel('objective height (mm)')
    plt.ylabel('atom waist (um)')    
    #plt.savefig(r'X:\rydberglab\Tweezers\2019\07\19\2019-07-18_fittedAtomSizeVsObj.png')