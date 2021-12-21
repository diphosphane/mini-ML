#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
various util function to use
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

import numpy as np
from typing import Dict
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    def fit(self, X, y): 
        if len(X)[0] != len(y)[0]:
            raise Error('Error, size of X and y is not equal.')
    
    @abstractmethod
    def predict(self, X): pass
    
    @staticmethod
    def mse(y_ref, y_pred) -> float:
        tmp = (y_ref - y_pred) ** 2
        return tmp.sum() / len(tmp)
    
    @abstractmethod
    def set_kw(self, kw: Dict[str, float]): pass


if __name__ == '__main__':
    print('you are running a inner lib, exiting')
