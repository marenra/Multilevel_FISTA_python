# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:44:42 2022

@author: Maren
"""
import numpy as np

'''
Construct a Gaussian blur point spread function

psfGauss(dim);
psfGauss(dim, s);

Input:
    dim     Desired dimension of the PSF array.
    s       Vector with standard deviations of the Gaussian along
            the vertical and horizontal directions.
            If s is a scalar then both standard deviations are s.
            Default is s = 2.0.

 Output:
     PSF     Array containing the point spread function.
     center  [row, col] gives index of center of PSF
'''
## in contrast to Matlab, Pythons indices start at 0 instead of 1,
## therefore the coordinates of center are reduced by 1

def psfGauss(dim,s = 2.0):
    l = np.size(dim)
    if l==1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]
    if np.size(s) == 1:
        s = np.array([s,s])
    #set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(n/2), np.ceil(n/2))
    y = np.arange(-np.fix(m/2), np.ceil(m/2))
    X,Y = np.meshgrid(x,y)
    
    #compute the gaussian and normalize the PSF
    PSF = np.exp(-(X**2)/(2*s[0]**2) - (Y**2)/(2*s[1]**2))
    PSF = PSF / PSF.sum()

    #define center point
    mm,nn = np.where(PSF == PSF.max())
    center = np.array([mm[0], nn[0]])
     
    return PSF, center

P, q = psfGauss(5)
print(P, q)



#PSF for Moffat blur used for astronomical telescopes
def psfMoffat(dim, beta, s = 2.0):
    l = np.size(dim)
    if l==1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]
    if np.size(s) == 1:
        s = np.array([s,s])
    #set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(n/2), np.ceil(n/2))
    y = np.arange(-np.fix(m/2), np.ceil(m/2))
    X,Y = np.meshgrid(x,y)
    
    #compute the gaussian and normalize the PSF
    PSF = (1+(X**2)/s[0] + (Y**2)/s[1])**(-beta)
    PSF = PSF / sum(PSF[ : ])

    #define center point
    mm,nn = np.where(PSF == PSF.max())
    center = np.array([mm[0], nn[0]])
     
    return PSF, center

#g,Q = psfMoffat(3, 2)
#print(g,Q)

