# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:44:19 2022

@author: Maren
"""
import numpy as np

"""
Pad a PSF array with zeros to make it bigger

padPSF(PSF, m) for quadratic output
padPSF(PSF, m, n)

Input:
    PSF     Array containing the point spread function
    m,n     Desired dimension of padded array
    
Output:
    P       Padded m-by-n array
"""

def padPSF(PSF, m, n = 'None'):
    if n == 'None':
        n = m
    P = np.zeros([m,n])
    P[0: np.shape(PSF)[0], 0: np.shape(PSF)[1]] = PSF
    return P
    