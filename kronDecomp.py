# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:12:18 2022

@author: Maren
"""
import numpy as np
import numpy.linalg as LA
from scipy.linalg import toeplitz
from scipy.linalg import hankel
import warnings

"""
Compute terms of Kronecker product factorization A = kron(Ar, Ac),
where A is a blurring matrix defined by a PSF array.  The result is
an approximation only, if the PSF array is not rank-one.

Input:
    P       Array containing the point spread function.
    center  [row, col] = indices of center of PSF, P.
    BC      String indicating boundary condition.
            ('zero', 'reflexive', or 'periodic')
            Default is 'zero'.

Output:
   Ac, Ar  Matrices in the Kronecker product decomposition.  
           Some notes:
             * If the PSF, P is not separable, a warning is displayed 
               indicating the decomposition is only an approximation.
             * The structure of Ac and Ar depends on the BC:
                 zero      ==> Toeplitz
                 reflexive ==> Toeplitz-plus-Hankel
                 periodic  ==> circulant 
"""

def kronDecomp(P, center, BC = 'zero'):
    # Get SVD decomposition
    [U, S,Vh] = LA.svd(P)
    V = Vh.T
    
    # Find the two largest singular values and corresponding singular 
    # vectors of the PSF -- these are used to see if the PSF is separable.
    if (S[1]/S[0] > np.sqrt(np.finfo(float).eps)):
        warnings.warn('THE PSF, P is not separable')
        
    # Check if the vectors of the rank-one decomposition 
    # of the PSF have nonnegative components. 
    minU = abs(min(U[:,0]))
    maxU = max(abs(U[:,0]))
    if minU == maxU:
        U = -U
        V = -V
    
    # Calculate the column and row vectors
    c = np.sqrt(S[0]*U[:,0])
    r = np.sqrt(S[0]*V[:,0])
    
    #build the Ar and Ac matrices 
    #depending on the imposed boundary condition
    if BC == 'zero':
        Ar = buildToep(r, center[1])
        Ac = buildToep(c, center[0])
    elif BC == 'refelxive':
        Ar = buildToep(r, center[1])+ buildHank(r, center[1])
        Ac = buildToep(c, center[0]) + buildHank(c, center[0])
    elif BC == 'periodic':
        Ar = buildCirc(r, center[1])
        Ac = buildCirc(c, center[0])
    return Ar, Ac
        

#help functions to construct the structure of the Ar and Ac matrices
'''
Build a banded Toeplitz matrix from a central column and an index
denoting the central column.
'''
def buildToep(c, k):
    # Toeplitz matrices
    n = np.size(c)
    col = np.zeros(n)
    row = np.zeros(n)
    col[0: n-k+1] = c[k-1:n]
    row[0:k] = c[0:k][::-1]
    return toeplitz(col,row)

'''
Build a banded circulant matrix from a central column and an index
denoting the central column.
'''
def buildCirc(c,k):
    n = np.size(c)
    col = np.append(c[k-1:n], c[0:k-1])
    row = np.append(c[0:k][::-1], c[k:n][::-1])
    return toeplitz(col,row)

'''
Build a Hankel matrix for separable PSF and reflexive boundary
conditions.
'''    
def buildHank(c,k):
    n = np.size(c)
    col = np.zeros(n)
    col[0:n-k] = c[k:n]
    row = np.zeros(n)
    row[n-k+1:n] = c[0:k-1]
    return hankel(col,row)
    
    
    