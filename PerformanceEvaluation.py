# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:25:08 2018

@author: user
"""

import numpy as np
from IrisMatching import match
#%%
def evaluation(l1, l2):
    return (np.sum(l1 == l2)/len(l2))*100

#%%
def graph(x_train, x_test, y_train, y_test):
    dim_fv = np.arange(10, 140, 10)
    rates = np.zeros(len(dim_fv))
    for i, v in enumerate(dim_fv):
        _, d3 = match(x_train, y_train, x_test, y_test, True, n_comp=v)
        rates[i] = (evaluation(d3[:,2], y_test))
    return rates

#%%
def falsematch(val, pred, thresh):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i, v in enumerate(val):
        if v < thresh:
            if pred[i] == (i//4)+1:
                TP += 1
            else:
                FP += 1
        else:
            if pred[i] == (i//4)+1:
                FN += 1
            else:
                TN += 1
    
    return (FP/(TP+FP) if TP+FP != 0 else 0, 
            FN/(TN+FN) if TN+FN != 0 else 0, 
            TP/(TP+FN) if TP+FN != 0 else 0, 
            FP/(FP+TN) if TP+FN != 0 else 0)
