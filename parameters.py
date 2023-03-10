#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ewaldow
"""

import math

# variables
L = 10**6 # distance (1000km)
B = 10 * 10**9 # bandwidth (1 to 10GHz)

# physical constants
c = 3 * 10**8 # speed of light
a_dB = 0.2 # power loss in dB [dB/km]
D = 17 # dispersion ps/(nm-km)
gamma = 1.27 # nonlinearity coefficient 1/(W km)
nsp = 1 # a constant factor
h = 6.626 * 10 ** (-34) # Planck constant J*s
lambda0 = 1.55 * 10 ** (-6) # center wavelength
f0 = c/lambda0 # center frequency
alpha = 10**(-4) * math.log10(a_dB) # loss coefficient
beta2 = -lambda0**2/(2*math.pi*c)*D # dispersion coefficient

# scale factors
L0 = L
T0 = math.sqrt(abs(beta2)*L/2)
P0 = 2/(gamma*L)

Pnorm = 0.006/P0

# noise PSD
sigma02 = nsp*h*alpha*f0 # physical
sigma2 = sigma02*L/(P0*T0) # normalized