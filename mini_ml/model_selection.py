#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
various util function to use
'''

# Author: diphosphane <diphosphane@gmail.com>
# License: GPL-3.0 License

import numpy as np
from typing import Dict, List, Any
from base import BaseEstimator
from kernel import Kernel as K


class GridSearch(BaseEstimator):
    def __init__(self, estimator: BaseEstimator, param_grid: Dict[str, Any], cv: int=0) -> None:
        super().__init__()
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_param: Dict[str, Any] = dict()
    
    def fit(self, X, y):
        train_idx_list, val_idx_list = self._cv_split(X.shape[0])
        best_mse = 1e50
        # best_model = None
        self.best_param: Dict[str, Any] = dict()
        keys, values = zip(*self.param_grid.items())
        for params_value in product(*values):
            params = dict(zip(keys, params_value))
            self.estimator.set_kw(params)
            mse = self._evaluate_model(self.estimator, X, y, train_idx_list, val_idx_list)
            if mse < best_mse:
                best_mse = mse
                self.best_param = params
                # best_model = deepcopy(self.estimator)
        self.estimator.set_kw(self.best_param)
        self.estimator.fit(X, y)
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def _cv_split(self, cnt: int):
        train_cnt = int(cnt * 0.8); # val_cnt = cnt - train_cnt
        train_idx_list = []; val_idx_list = []
        if self.cv in [0, 1]:
            train_idx_list.append(np.arange(train_cnt))
            val_idx_list.append(np.arange(train_cnt, cnt))
        else:
            batch_num = cnt // self.cv
            for i in range(self.cv):
                flg1, flg2 = batch_num * i, batch_num * (i+1)
                train_idx_list.append(np.r_[np.arange(flg1), np.arange(flg2, cnt)])
                val_idx_list.append(np.arange(flg1, flg2))
        return train_idx_list, val_idx_list
        
    @classmethod
    def _evaluate_model(cls, estimator: BaseEstimator, X, y, train_idx_list: List[Any], val_idx_list: List[Any]):
        mse_list = []
        for train_idx, val_idx in zip(train_idx_list, val_idx_list):
            estimator.fit(X[train_idx], y[train_idx])
            y_pred = estimator.predict(X[val_idx])
            mse = cls.mse(y[val_idx], y_pred)
            mse_list.append(mse)
        return np.array(mse_list).mean()
    
    def set_kw(self, kw: Dict[str, float]): pass


if __name__ == '__main__':
    print('you are running a inner lib, exiting')
