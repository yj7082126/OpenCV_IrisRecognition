# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:25:08 2018

@author: user
"""

import numpy as np

#%%
def evaluation(l1, l2):
    return (np.sum(l1 == l2)/len(l2))*100