#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ewaldow
"""

import numpy as np
import matplotlib.pyplot as plt
from generative import (
        cnt, source, bit_to_symb, 
        mod, channel, equalize, compare, 
        symb_to_bit, detector, demod);
from parameters import B, T0

"""
b = source(128, 1/2)
s = bit_to_symb(b, cnt)

T = 400; # you have to choose this
N = 2**10; # you have to choose this
dt = T/N;
t = np.arange(-T/2,T/2,dt)

z = 0 # should go from 0 to 1 (normalization)

q0t = mod(t,s,B)
[qzt,qzf] = channel(t, q0t, z, 0, 0)

[qzte, qzfe] = equalize(t, qzt, z)
print(compare(qzte,q0t)) # should be equal when either z = 0 or B = 0 on absence of noise
"""

###

def test_generative_functions():
    T = 400;
    N = 2**10;
    dt = T/N;
    t = np.arange(-T/2,T/2,dt)
    #B = 1*10**9
    
    b = source(128,1/2)
    s = bit_to_symb(b,cnt)
    bhat0 = symb_to_bit(s,cnt)
    if np.sum(b!=bhat0) :
        print("ERROR in bit_to_symb or symb_to_bit")
        return
    
    #print("bit_to_symb and symb_to_bit working as expected :)")
    stilde1 = detector(s, cnt)
    bhat1 = symb_to_bit(stilde1, cnt)
    if np.sum(b!=bhat1) :
        print("ERROR in detector")
        return
    
    #print("detector working as expected :)")
    q0t = mod(t, s, B)
    Ns = len(s)
    shat = demod(t, q0t, B, Ns)
    stilde2 = detector(shat, cnt)
    bhat2 = symb_to_bit(stilde2, cnt)
    if np.sum(b!=bhat2) :
        print("ERROR in mod or demod, "+str(np.sum(b!=bhat2))+"/"+str(len(b))+" errors in b/bhat")
        return
    
    print("all tests passed :)")

def plot_q0t():
    T = 400;
    N = 2**10;
    dt = T/N;
    t = np.arange(-T/2,T/2,dt)
    #B = 1*10**9
    
    # normalize ?
    t = t/T0
    
    b = source(128,1/2)
    s = bit_to_symb(b,cnt)
    q0t = mod(t, s, B*T0)
    
    fig, axs = plt.subplots(2)
    fig.suptitle('q0t magnitude and argument')
    axs[0].plot(t,np.abs(q0t))
    axs[1].plot(t,np.angle(q0t))
    
    p_q0t = np.sum(np.abs(q0t**2))
    p_q0t_avg = p_q0t/len(q0t)
    
    print("average q0t power = "+str(p_q0t_avg))
    print("total q0t energy = "+str(p_q0t))

def plot_cnt():
    plt.scatter(np.real(cnt), np.imag(cnt))