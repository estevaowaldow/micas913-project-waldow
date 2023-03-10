#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ewaldow
"""

import numpy as np
from generative import source, bit_to_symb, mod

# QUESTION 7, NOT DONE
# time mesh
T = 400; # you have to choose this
N = 2**10; # you have to choose this
dt = T/N;
t = np.arange(-T/2,T/2,dt) # should be from -T/2 to T/2 with length N

# frequency mesh
F = 1/dt;
df = 1/T;
f = np.arange(-F/2,F/2,df) # should be from -F/2 to F/2 with length N

# bits to signal
M = 16; # size of the constellation
n = 3; # number of symbols (or sinc functions); test with s=1
nb = ... ; # number of bits
p =1/2; # probability of zero

b = source(nb, p); # Bernoulli source, random bits sequence
s = bit_to_symb(b, cnt); # symbol sequence; could be inside modulator.m
q0 = mod(t, s, B); # transmitted signal




# QUESTION 14, NOT DONE
A = 1;
q0t = A*exp(-t**2); # Gaussian input, for testing
q0f = ... # input in frequency
# plot below the input in t & f. You must tune T and N!
...
...

# propagation
[qzt, qzf] = channel(t, q0t, z, 0, 0); # output in t,f. Zero noise.
plot(t, q0t, ...) # plot input output in t; tune T and N accordingly
plot(f, abs(q0f), ...) # plot input output in f; tune T and N accordingly

# equalization and comparison
[qzte, qzfe] = equalize(t, qzt, z); # equalized output
plot(t, q0t, ...) # plot input & equalized output in t
compare(q0t, qzte) # you should write the function compare


# QUESTION 17, NOT DONE
# modulation
nb = 64; # number of bits
M = 16; # order of modulation
ns = ... # number of symbols
b= ... # random bit sequence
s = bit_to_symb(b, cnt);
q0t = mod(t, s, B);

# propagation & equalization. Set the noise to zero for now
[qzt, qzf] = channel(t, q0t, z, sigma2, B); # output in t,f
[qzte, qzfe] = equalize(t, qzt, z); # equalized output

# demodulation
shat = demod(t, qzte, B);

# detection
stilde = detector(shat, cnt);
bhat = symb_to_bit(stilde, cnt);