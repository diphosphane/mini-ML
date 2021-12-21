#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
various util function to use
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

import numpy as np
from typing import Dict
from base import BaseEstimator
from kernel import Kernel as K


class KRR(BaseEstimator):
    def __init__(self, kernel=K.Linear, lambda_=0.1, **kw):
        self.kernel = getattr(self, kernel)
        self.params = {'lambda': lambda_, 'sigma': 0.01}
        self.set_kw(kw)
        self.train_x = None
        self.alpha = None
    
    def set_kw(self, kw: Dict[str, Any]):
        for k, v in kw.items():
            self.params[k] = v
    
    def Linear(self, x1, x2):
        return x1 @ x2.T
    
    def Gaussian(self, x1, x2):
        dist = np.sum(x1**2, 1).reshape(-1, 1) \
               + np.sum(x2**2, 1) \
               - 2*np.dot(x1, x2.T)
        return np.exp(-0.5 * dist / (self.params['sigma']**2))
    
    def fit(self, X, y):
        self.train_x = np.asarray(X)
        self.train_y = np.asarray(y).reshape(-1, 1)
        K = self.kernel(self.train_x, self.train_x)
        self.alpha = np.linalg.inv(K + self.params['lambda'] * np.eye(len(self.train_x))) @ self.train_y
    
    def predict(self, X):
        self.pred_x = np.asarray(X)
        K = self.kernel(self.pred_x, self.train_x)
        return K @ self.alpha

if __name__ == '__main__':
    print('you are running a inner lib, exiting')
