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
from kernel import BaseKernel


class KRR(BaseEstimator):
    def __init__(self, kernel: BaseKernel=K.Linear, lambda_=0.1, **kw):
        self.params = {'lambda': lambda_, 'sigma': 0.01}
        self.set_kw(kw)
        self.kernel = kernel; self.kernel.set_params(self.params)
        self.train_x = None
        self.alpha = None
    
    def set_kw(self, kw: Dict[str, Any]):
        self.kernel.set_params(kw)
        for k, v in kw.items():
            self.params[k] = v
    
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
