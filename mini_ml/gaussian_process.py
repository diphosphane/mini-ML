#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
various util function to use
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

from mini_ml.kernel import BaseKernel
import numpy as np
from typing import Dict
from base import BaseEstimator
from kernel import Kernel as K


class GPR(BaseEstimator):
    def __init__(self, kernel: BaseKernel=K.RBF, optimizer='L-BFGS-B', opt_params=None, bounds=None, **kw):
        self.optimizer = optimizer           # optimizer defined in scipy.optimize.minize
        self.params = {                      # initial parameter.
            "l": 0.5,       # l: length_scale in kernel formula
            "sigma": 0.2,   # sigma: σ in kernel
            "alpha": 1e-8   # alpha: σ in cov function
        }
        self.opt_params = opt_params  # a list/tuple with the name of the parameter needs to be optimized, e.g. ['l', 'sigma']
        self.bounds = bounds          # the lower and upper bound with the same order to opt_params, e.g. [(1e-4. 1e4), (1e-5, 1e3)]
        self.set_kw(kw)
        self.is_fit = False           # fit flag
        self.kernel = kernel; self.kernel.set_params(self.params)
        self.train_X, self.train_y = None, None  # var to store training data
    
    def set_kw(self, kw: Dict[str, Any]):
        self.kernel.set_params(self.params)
        for k, v in kw.items():
            self.params[k] = v

    def predict(self, X, return_std=False):
        X = np.asarray(X)    # convert to numpy object
        # if not fited, return the prior distribution
        if not self.is_fit:  
            mean = np.zeros(X.shape[0])   # mean = 0.0  size: (N,)
            cov = self.kernel(X, X)       # cov = K(x_train, x_train)
            if return_std:   # return std instead of cov
                return mean, np.sqrt(np.diag(cov))
            return mean, cov
        # if fitted, make prediction  
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + self.params["alpha"] * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y).ravel()  # mean fuction
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)            # cov function
        if return_std:    # return std instead of cov
            return mu, np.sqrt(np.diag(cov))
        return mu, cov

    def sample_y(self, X, n_samples=5, random_state=0):   # sample many y curve with "n_samples" number and random seed "random_state"
        mean, cov = self.predict(X[:, np.newaxis])
        rs = np.random.RandomState(random_state)
        return rs.multivariate_normal(mean, cov, n_samples).T  # generate many y curve with multivariate normal method
    
    def negative_log_likelihood_loss(self, params):  # function to be minized
        for k, v in zip(self.opt_params, params):    # set the trial hyperparameter to GPR instance
            self.params[k] = v
        Kyy = self.kernel(self.train_X, self.train_X) + self.params["alpha"] * np.eye(len(self.train_X))
        loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + \
               0.5 * np.linalg.slogdet(Kyy)[1] + \
               0.5 * len(self.train_X) * np.log(2 * np.pi)
        return loss.ravel()
    
    def fit(self, X, y):
        # convert to a numpy object
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # hyper parameters optimization
        if self.optimizer and self.opt_params:
            res = minimize(self.negative_log_likelihood_loss, 
                           [self.params[k] for k in self.opt_params],  # initial guess of hyperparameter that is need to be optimized
                           bounds=self.bounds,     # bound of hyperparameter
                           method=self.optimizer)  # name of optimizer in scipy.optimize.minize
            for i, k in enumerate(self.opt_params):  # finally set the optimized hyperparameter
                self.params[k] = res.x[i]
        self.is_fit = True

if __name__ == '__main__':
    print('you are running a inner lib, exiting')
