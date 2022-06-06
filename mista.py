# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:21:52 2022

@author: Maren
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from ista import ista
from psfGauss import psfGauss
'''
This function provides top-level parameters for multilevel ISTA/FISTA to
solve the composite convex optimisation. 

% INPUT
%
% inputImage    : Filename of the original image
% inputMethod   : 'ista' or 'fista'
% pars          : Parameters structure
% pars.MAXITER  : maximum number of iterations (Default = 50)
% pars.lambda   : regulariser parameter (1e-5)
% pars.wave     : orthogonal transformation (@dct)
% pars.iwave    : orthogonal transformation (@idct)
% pars.dim      : psf window ([16,16])
% pars.s        : psf standard deviation (8)
% pars.noisy    : Guassian Noise added to image (0.001)
% ** larger dim,s,noisy produces more corrupted images 
% pars.fig      : if fig>0 show image at each iteration, 0 otherwise (0)
% pars.BC       : Boundary Condition 'reflexive'(default) or 'periodic'
% pars.epsilon  : stopping condition, the percentage of F_k/F_0 (0.02)
% ** Without epsilon, program will run MAXITER iterations
% pars.kappa    : first condition to do coarse correction (0.49)
% pars.eta      : second condition to do coarse correction (1) 
% ** Smaller kappa & eta encourage more coarse correction
% -----------------------------------------------------------------------
% OUTPUT
% 
% X             : Solution of the problem min |A(X)-B|^2 + lambda |Wx|_1
% f             : history of objective functions
'''

pars = {'MAXITER' : 50, 'wave': dct , 'iwave' : idct, 'fig': 0, 'mu' : 0.2, 'epsilon' : 0.02, 'BC' : 'reflexive', 'lam': 1e-5, 'eta': 1, 'kappa' : 0.49, 'dim' : [16,16], 's': 8, 'noisy': 0.001}
inputImage = 'dog.jpeg'

def mista(inputImage, inputMethod, pars):
    X = plt.imread(inputImage)
    minX = X.min()
    maxX = X.max()
    X = (X - minX) / (maxX - minX)
    
    P, center = psfGauss(pars['dim'],pars['s'])
    
    #scipymisc.imfilter evtl für imfilter
    B = inputImage #but blurred version
    
    fh = globals()[inputMethod]
    X, f, coarse_iters = fh(B,P,center,pars['lam'],pars)

    #PLOTS
    
    #return X, f
