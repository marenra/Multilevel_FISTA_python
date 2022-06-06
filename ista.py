# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:11:13 2022

@author: Maren
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
#from scipy.fft import dct, idct
#from scipy.sparse import coo_matrix
from scipy.linalg import svdvals

from padPSF import padPSF
from kronDecomp import kronDecomp
from smooth_sqrt import smooth_sqrt

'''
This function implements Multilevel ISTA for solving the linear inverse
problem with an orthogonal transform regulariser using cosine transformation 
and "reflexive" boundary condition. This function performs computations 
at the finest resolution, and it calls smooth_sqrt function to obtain
coarse error correction. It is also possible to use this function
recursively to obtain coarse error correction.

% INPUT
%
% B             : The observed image which is blurred and noisy
% P             : point-spread-function PSF of the blurring operator
% center        : A vector of length 2 containing the center of the PSF
% lambda        : Regularisation parameter
% ** P & center for computation of sparse column/row blurring operators
% pars          : Parameters structure
% pars.MAXITER  : maximum number of iterations (Default = 50)
% pars.wave     : orthogonal transformation (@dct)
% pars.iwave    : orthogonal transformation (@idct)
% pars.fig      : if fig>0 show image at each iteration, 0 otherwise (7)
% pars.start    : starting image for optimisation
% pars.epsilon  : stopping condition, the percentage of F_k/F_0 (0.02)
% ** Without epsilon, program will run MAXITER iterations
% pars.kappa    : first condition to do coarse correction (0.49)
% pars.eta      : second condition to do coarse correction (1) 
% ** Smaller kappa & eta encourage more coarse correction
% -----------------------------------------------------------------------
% OUTPUT
% 
% X_iter        : Solution of the problem min |A(X)-B|^2 + lambda |Wx|_1
% fun_all       : History of function values at the finest resolution
% smooth_iter   : Record coarse correction iterations

'''

def ista(B, P, center, lam, pars):
    W = pars['wave']
    WT = pars['iwave']
    B_start = B
    MAXITER = pars['MAXITER']
    fig = pars['fig']
    BC = pars['BC']
    kappa = pars['kappa']
    eta = pars['eta']
    epsilon = pars['epsilon']
    #Initialization for fine level
    m,n = np.shape(B)
    Pbig = padPSF(P, m, n)
    Ar, Ac = kronDecomp(Pbig, center, BC)
    Ar[Ar<1e-10] = 0
    Ac[Ac<1e-10] = 0
    #indices are different to matlab!
    #Ar = coo_matrix(Ar)
    #Ac = coo_matrix(Ac)
    sr = svdvals(Ar)
    sc = svdvals(Ac)
    Sbig = np.kron(sr, sc).reshape([len(sr), len(sc)])
    #Lipschitz constant of the gradient of |A(X)-B|^2
    L = 2 * max(max(abs(Sbig)**2))
    #starting point
    X_iter = B_start
    #compute first function value
    t = sum(sum(abs(W(X_iter))))
    fun_all = norm(Ac @ X_iter @ Ar.T - B, 'fro')**2 + lam*t
    
    ## Parameters for coarse levels
    m,n = np.shape(B)
    M = m/2
    N = n/2
    id1 = [x for x in range(m) if x % 2 != 0]
    id2 = [x for x in range(1, m) if x % 2 == 0]
    for i in range(1,n/2):
        id1 = [id1, [x for x in range(2*i*m+1,2*i*m+m) if x% 2 != 0]]
        id2 = [id2, [x for x in range(2*i*m+2,2*i*m+m) if x% 2 == 0]]
    smallB = B(id1) + B(id2) + B(id1+m) + B(id2+m)
    smallB = (smallB/4).reshape([N,M]).T
    smallPars = pars
    smallPars['epsilon'] = 0.01
    
    R = np.ones(2)
    R = np.kron(np.eye(M), R)
    Ac_H = R @ Ac @ R.T/2
    Ar_H = R @ Ar @ R.T/2
    smooth_iter = []
    
    ##Start fine iterations
    X_lastcoarse = np.random.rand(np.shape(X_iter))
    
    print('********************MISTA-F **********************\n')
    print('#iter  funtion     kappa   eta -------------------\n')
    print('%4d  %10.5f\n',0,fun_all);
    l = 0
    for i in range(1, MAXITER +1):
        X_old = X_iter
        X_iter = (X_old-2/L) @ Ac.T @ (Ac @ X_old @ Ar.T -B) @ Ar
        WX = W(X_iter)
        D = abs(WX) - lam/L
        WX = np.sign(WX) * D # actually ((D>0).*D)
        X_iter = WT(WX)
        d_h = X_old - X_iter
        D_H = d_h[id1] + d_h[id2] + d_h[id1+m] + d_h[id2+m]
        D_H = (D_H/4).reshape([[n,M]]).T
        R = X_old[id1] + X_old[id2] + X_old[id1+m] + X_old[id2+m]
        smallB_start = (R/4).reshape
        #Compute coarse correction conditions
        nD_h = norm(X_old - X_lastcoarse, 'fro')
        nGrad = norm(d_h, 'fro')
        nD_H = norm(D_H, 'fro')/nGrad
        if ((nD_H > kappa) and (nD_h > eta) and (nGrad > eta)):
            #coarse error correction
            R = smooth_sqrt(smallB, smallB_start, lam/2, smallPars, D_H, Ac_H, Ar_H, l)
            d_h = np.kron(R - smallB_start, np.ones([2,2]))
            tau = 1
            l = l+1
            smooth_iter = [smooth_iter, i]
        else:
            #Gradient step
            tau = 1/L
            d_h = -2 * Ac.T @ (Ac @ X_old @ Ar.T -B) @ Ar
        
        X_iter = X_old + tau *d_h
        WX = W(X_iter)
        D = abs(WX)-lam*tau
        WX = np.sign(WX) * D
        X_iter = WT(WX)
        t = sum(sum(abs(WX)))
        fun_val = norm(Ac @ X_iter @ Ar.T -B, 'fro')**2 + lam*t
        
        fun_all = [fun_all, fun_val]
        
        if fig > 0:
            plt.plot(fig)
            plt.imshow(X_iter)
        
        print('%4d  %10.5f  %6.3f  %6.3f',i,fun_val,nD_H,nGrad)
        if ((nD_H > kappa) and (nD_h > eta) and (nGrad > eta)):
            print('coarse correction')
            X_lastcoarse = X_old
            
        print('\n')
        if fun_all[-1]/fun_all[0] < epsilon:
            break
        
    return X_iter, fun_all, smooth_iter

#pars = {'MAXITER' : 50, 'wave': dct , 'iwave' : idct, 'fig': 0, 'mu' : 0.2, 'epsilon' : 0.02, 'BC' : 'reflexive', 'lam': 1e-5, 'eta': 1, 'dim' : [16,16], 's': 8, 'noisy': 0.001}
