#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
defined the common used kernel
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class Kernel():
    Linear = 'Linear'
    Gaussian = 'Gaussian'
    RBF = 'RBF'
    Exponential = 'Exponential'
    Matern = 'Matern'
    RationalQuadratic = 'RationalQuadratic'

class BaseKernel(ABC):
    params: Dict[str, float] = {}
    @abstractmethod
    def __call__(self, x1, x2) -> Any: pass

    def set_params(self, params: Dict[str, float]):
        self.params = params
    

class Linear(BaseKernel):
    def __call__(self, x1, x2):
        return x1 @ x2.T


class Gaussian(BaseKernel):
    def __call__(self, x1, x2) -> Any:
        dist = np.sum(x1**2, 1).reshape(-1, 1) \
               + np.sum(x2**2, 1) \
               - 2*np.dot(x1, x2.T)
        return np.exp(-0.5 * dist / (self.params['sigma']**2))

class RBF(BaseKernel):
    def __call__(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)        

class Exponential(BaseKernel) :
    def __call__(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        dist_matrix = dist_matrix ** 0.5
        return self.params['sigma']**2 * np.exp(-1/self.params['l'] * dist_matrix)

class Matern(BaseKernel) :
    def __call__(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        dist_matrix = dist_matrix ** 0.5
        tmp1 = 1 + 3**0.5 * dist_matrix / self.params['l']
        tmp2 = np.exp(-3**0.5 * dist_matrix / self.params['l'])
        return self.params['sigma'] * tmp1 * tmp2
    
class RationalQuadratic(BaseKernel):
    def __call__(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        alpha = self.params['kernel_alpha']
        l = self.params['l']
        inner = 1 + dist_matrix / (2 * alpha * l**2)
        return self.params['sigma']**2 * inner**(-alpha)


if __name__ == '__main__':
    print('you are running a inner lib, exiting')
