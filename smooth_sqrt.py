# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:47:33 2022

@author: Maren
"""

import numpy as np
from numpy.linalg import norm
from scipy.fft import dct, idct

'''
% INPUT
%
% B             : The coarse corrupted image
% B_start       : starting image for optimisation
% lambda        : Regularisation parameter
% pars          : Parameters dictionary
% pars.MAXITER  : maximum number of iterations (Default = 50)
% pars.wave     : orthogonal transformation (@dct)
% pars.iwave    : orthogonal transformation (@idct)
% pars.epsilon  : stopping condition, the changes |F_k - F_(k+1)| (0.01)
% v             : linear term for first order coherence (see Parpas et.al.)
% Ac,Ar         : column/row blurring operators
% l             : condition to do coarser levels

% OUTPUT
% 
% X_iter        : Solution of the problem min |A(X)-B|^2 + lambda |Wx|_1
% fun_all       : History of function values at the finest resolution

'''
pars = {'MAXITER' : 100, 'wave': dct , 'iwave' : idct, 'mu' : 0.2, 'epsilon' : 0.02}

def smooth_sqrt(B, B_start, lam, pars, v, Ac, Ar, l):
    MAXITER = pars['MAXITER']
    W = pars['wave']
    WT = pars['iwave']
    mu = pars['mu']
    epsilon = pars['epsilon']
    
    e = 0.01
    m,n = np.shape(B)
    
    X_iter = B_start
    WX = W(X_iter)
    tmp = np.sqrt(WX**2 + mu**2)
    g = (Ac @ X_iter @ Ar.T) - B
    fun_val = norm(g, 'fro')**2 + lam * (sum(tmp)-m*n*mu)
    if np.size(v) > 1:
        #get the residual v term 
        v1 = v - 2*g
        v2 = -lam * 0.5 * 1/tmp
        fun_all = fun_val
        if l == 0:
            M = m/2
            N = n/2
            R = np.kron(np.eye(M,M), np.ones(2))
            Ac_H = R @ Ac @ R.T /2
            Ar_H = R @ Ar @ R.T /2
            id1 = [x for x in range(m) if x % 2 != 0]
            id2 = [x for x in range(1, m) if x % 2 == 0]
            for i in range(1,n/2):
                id1 = [id1, [x for x in range(2*i*m+1,2*i*m+m) if x% 2 != 0]]
                id2 = [id2, [x for x in range(2*i*m+2,2*i*m+m) if x% 2 == 0]]
            R = X_iter(id1) + X_iter(id2) + X_iter(id1+m)+ X_iter(id2+m)
            smallB_start = (R/4).reshape((N,M)).T
            smallPars = { 'epsilon' : epsilon}
            R = B(id1) + B(id2) + B(id1+m) + B(id2+m)
            R = smooth_sqrt((R/4).reshape((N,M)).T, smallB_start, lam/2, smallPars, 0, Ac_H, Ar_H, 1)
            X_iter = X_iter + np.kron(R-smallB_start, np.ones(2,2))
            WX = W(X_iter)
            tmp = np.sqrt(WX**2 + mu**2)
            g = (Ac @ X_iter @ Ar.T) - B
            fun_val = norm(g, 'fro')**2 + lam * (sum(tmp)-m*n*mu)
            fun_all = [fun_all, fun_val]
            if fun_all[-1]/fun_all[0]< epsilon:
                return X_iter, fun_all
    else:
        v1 = 0
        v2 = 0
        fun_all = fun_val
    
    g = Ac.T @ g @ Ar
    for i in range(1, MAXITER+1):
        X_old = X_iter
        t = 1
        g = -2*g- e*v1
        normg = norm(g, 'fro')**2
        fun_val = fun_all[-1] + 1000
        while fun_val > fun_all[-1]+ 0.01 *t*normg:
            X_iter = X_old + t*g
            WX = W(X_iter)
            WX = WX - lam/t*0.5*tmp - e/t*v2
            X_iter = WT(WX)
            tmp = np.sqrt(WX**2 + mu**2)
            g = (Ac @ X_iter @ Ar.T) - B
            fun_val = norm(g, 'fro')**2 + lam*(sum(tmp)-m*n*mu)
            t = t/2
        fun_all = [fun_all, fun_val]
        g = Ac.T @ g @ Ar
        if abs(fun_all[-2]- fun_all[-1]) < epsilon:
            break
    return X_iter, fun_all
            
        
    
    