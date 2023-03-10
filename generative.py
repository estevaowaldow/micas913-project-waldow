#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ewaldow
"""

import math
import numpy as np

from parameters import Pnorm

cnt = np.sqrt(Pnorm/10)*np.array(
        [-3-3j,-3-1j,-3+3j,-3+1j,
         -1-3j,-1-1j,-1+3j,-1+1j,
         3-3j,3-1j,3+3j,3+1j,
         1-3j,1-1j,1+3j,1+1j])
"""
cnt = np.sqrt(0.381)*np.array(
        [[-3,-3],[-3,-1],[-3,3],[-3,1],
         [-1,-3],[-1,-1],[-1,3],[-1,1],
         [3,-3],[3,-1],[3,3],[3,1],
         [1,-3],[1,-1],[1,3],[1,1]])
"""

# Question 5
def source(N,p):
    b = 1*(np.random.random(N)>p)
    return b

# Question 6
def bit_to_symb(b, cnt):
    N = len(b)
    M = len(cnt)
    logM = int(np.log2(M))
    n = int(N/logM)
    
    aux_bits_pow = 2**np.arange(logM)
    
    #if (np.size(np.shape(cnt)) != 1):
    #    s = np.zeros([n,np.shape(cnt)[1]],dtype = cnt.dtype)
    #else:
    s = np.zeros(n, dtype = 'complex')
    #sshape1 = np.shape(cnt)[1] if (np.size(np.shape(cnt)) != 1) else 1
    #s = np.zeros([n,sshape1],dtype = cnt.dtype)
    
    for i in range(n):
        subseq = b[i*logM : (i+1)*logM]
        s[i] = cnt[int(np.sum(subseq * aux_bits_pow))]
    
    return s

# Question 7
# WRONG
def mod(t, s, B):
    #q(t,0)=sum_{l=l1}^{l2} sl * sinc(Bt-l)
    Ns = len(s) # number of symbols
    l1 = -math.floor(Ns/2)
    l2 = math.ceil(Ns/2)-1
    
    aux = np.roll(s, l1) # same as np.concatenate((s[l1:], s[:l2+1]))
    l = np.arange(l1,l2+1)    
    
    q0t = np.zeros(len(t), dtype=s.dtype)
    for i in range(len(t)):
        q0t[i] = np.sum(aux * np.sinc(B*t[i] - l))

    return q0t

# Question 12
def channel(t, q0t, z, sigma2, B):
    a = sigma2*B*z # total noise power in B Hz and distance [0, z]
    
    N = len(t)
    T = -2*t[0]
    F = N/T
    df = 1/T
    f = np.arange(-F/2,F/2,df) # get the f vector from t vector
    
    # Shift ?
    q0f = np.fft.fft(q0t) # input in frequency
    
    hzf = np.exp(1j*(2*np.pi*f)**2*z) # channel impulse response in frequency
    qzf = q0f * hzf # output in frequency
    
    # add Guassian noise in frequency, with correct variance
    mean = [0,0] # is it ?
    cov = a/2*np.array([[1,0],[0,1]]) # is it ?
    noise = np.random.multivariate_normal(mean, cov, N)
    qzf = qzf + noise[:,0] + 1j*noise[:,1]
    
    # Shift ?
    qzt = np.fft.ifft(qzf) # back to time
    
    return [qzt, qzf]

# Question 14
def equalize(t, qzt, z):
    N = len(t)
    T = -2*t[0]
    F = N/T
    df = 1/T
    f = np.arange(-F/2,F/2,df) # get the f vector from t vector
    
    # Shift ?
    qzf = np.fft.fft(qzt) # input in frequency
    
    hzf = np.exp(1j*(2*np.pi*f)**2*z) # channel impulse response in frequency
    qzfe = qzf / hzf # output in frequency
    qzte = np.fft.ifft(qzfe) # back to time
    return [qzte, qzfe]

def compare(a,b):
    return np.corrcoef(a,b)

# Question 15
def demod(t, qzte, B, Ns):
    # sl = B * int_{-inf}^{inf} qe(t,L)sinc(Bt-l)
    shat = np.zeros(Ns, dtype='complex')
    
    for l in range(Ns):
        shat[l] = B*np.sum(qzte*np.sinc(B*t-l))

    return shat

# Question 16
def detector(shat,cnt):
    # Cheating ? I'm considering we already know cnt
    
    real_shat = np.real(shat)
    imag_shat = np.imag(shat)
    
    real_stilde = 0
    imag_stilde = 0
    
    stilde = np.zeros(len(shat), dtype='complex')
    
    for i in range(len(shat)):
        if real_shat[i] < 0:
            if real_shat[i] < -np.sqrt(Pnorm/10)*2:
                real_stilde = -np.sqrt(Pnorm/10)*3
            else:
                real_stilde = -np.sqrt(Pnorm/10)
        else:
            if real_shat[i] < np.sqrt(Pnorm/10)*2:
                real_stilde = np.sqrt(Pnorm/10)
            else:
                real_stilde = 3*np.sqrt(Pnorm/10)
        if imag_shat[i] < 0:
            if imag_shat[i] < -2*np.sqrt(Pnorm/10):
                imag_stilde = -3*np.sqrt(Pnorm/10)
            else:
                imag_stilde = -np.sqrt(Pnorm/10)
        else:
            if imag_shat[i] < 2*np.sqrt(Pnorm/10):
                imag_stilde = np.sqrt(Pnorm/10)
            else:
                imag_stilde = 3*np.sqrt(Pnorm/10)
        stilde[i] = real_stilde + 1j * imag_stilde
        
    return stilde

# Question 16
# SLOW
def symb_to_bit(stilde, cnt):
    n = len(stilde)
    M = len(cnt)
    logM = int(np.log2(M))
    N = n*logM
    
    bhat = np.zeros(N, dtype=int)
    
    for i in range(n):
        xhat = np.where(stilde[i]==cnt)[0][0]
        bhat[4*i:4*i+4] = np.array([xhat%2,xhat/2%2,xhat/4%2,xhat/8],dtype=int)
    
    return bhat

# Question 20
def noise(n, sigma2):
    Zcart = np.random.multivariate_normal([0,0], sigma2*np.identity(2), n)
    Z = Zcart[:,0] + 1j*Zcart[:,1]
    return Z

# Question 21
# NOT FINISHED
def nnet_gen(x, nz, params):
    # pre-compute the fixed weight matrix W
    n = len(x)
    f = ... # frequency vector
    w = 2*np.pi*f # angular frequency vector
    
    z = ... #params ...
    dz = z/nz # epsilon
    h = exp(j*w**2*dz); # all-pass filter
    D = ... # DFT matrix of size n
    W = np.conjugate(D) #* ...
    
    # Loop over the nnet layers
    for k in range(nz):
        # linear transformation -- multiplication by the weight matrix W
        x = np.matmul(W, x)
        
        # activation function
        x = sigma(x, dz)
        
        # noise addition
        x = x + noise(n, sigma2)
    y = x
    
    return y
