# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:52:00 2018

@author: user
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

#%%
def match(x_train, y_train, x_test, y_test, reduction, n_comp=120):
    if reduction:
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        pca = PCA(n_components=n_comp).fit(x_train)
        x_train_red = pca.transform(x_train)
        x_test_red = pca.transform(x_test)
        clf = LDA().fit(x_train_red, y_train)
    else:
        x_train_red = x_train
        x_test_red = x_test

    [n1,m1] = x_train_red.shape
    [n2,m2] = x_test_red.shape
    [n, m] = x_train.shape
    
    l = len(np.unique(y_train))
    fi=np.zeros((l,m1))
    
    for i in range(l):
        group = x_train_red[list(np.where(y_train==i+1)),:][0]
        fi[i,:]=(np.mean(group, axis=0))
    
    if reduction:
        x_test_red = clf.transform(x_test_red)
        fi = clf.transform(fi)
        
    d1 = np.zeros((n2,l))
    d2 = np.zeros((n2,l))
    d3 = np.zeros((n2,l))
    
    values_y = np.zeros((n2, 3))
    pred_y = np.zeros((n2, 3))
    for i in range(n2):
        for j in range(l):
            d1[i,j] = sum(abs((x_test_red[i,:]-fi[j,:])))
            d2[i,j] = sum((x_test_red[i,:]-fi[j,:])**2);              
            d3[i,j] = 1-(np.dot(x_test_red[i,:].T, fi[j,:]))/(np.linalg.norm(x_test_red[i,:])*np.linalg.norm(fi[j,:]))
         
        values_y[i, 0] = np.min(d1[i,:])
        values_y[i, 1] = np.min(d2[i,:])
        values_y[i, 2] = np.min(d3[i,:])
        pred_y[i, 0] = np.argmin(d1[i,:])+1
        pred_y[i, 1] = np.argmin(d2[i,:])+1
        pred_y[i, 2] = np.argmin(d3[i,:])+1
        
    return values_y, pred_y

        
