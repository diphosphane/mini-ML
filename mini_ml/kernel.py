#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
defined the common used kernel
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

import numpy as np
import enum
from dataclasses import dataclass
from typing import Dict

class Kernel():
    Gaussian = 'Gaussian'
    RBF = 'RBF'
    Exponential = 'Exponential'
    Matern = 'Matern'
    RationalQuadratic = 'RationalQuadratic'


def Gaussian(x1, x2, params: Dict[str, float]):
    dist = np.sum(x1**2, 1).reshape(-1, 1) \
            + np.sum(x2**2, 1) \
            - 2*np.dot(x1, x2.T)
    return np.exp(-0.5 * dist / (self.params['sigma']**2))

def RBF(x1, x2, params: Dict[str, float]):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return params["sigma"] ** 2 * np.exp(-0.5 / params["l"] ** 2 * dist_matrix)        


def Exponential(x1, x2, params: Dict[str, float]):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    dist_matrix = dist_matrix ** 0.5
    return params['sigma']**2 * np.exp(-1/params['l'] * dist_matrix)
    
def Matern(x1, x2, params: Dict[str, float]):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    dist_matrix = dist_matrix ** 0.5
    tmp1 = 1 + 3**0.5 * dist_matrix / params['l']
    tmp2 = np.exp(-3**0.5 * dist_matrix / params['l'])
    return params['sigma'] * tmp1 * tmp2

def RationalQuadratic(x1, x2, params: Dict[str, float]):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    alpha = params['kernel_alpha']
    l = params['l']
    inner = 1 + dist_matrix / (2 * alpha * l**2)
    return params['sigma']**2 * inner**(-alpha)

if __name__ == '__main__':
    print('you are running a inner lib, exiting')
