# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:25:08 2018

@author: user
"""

import numpy as np
from IrisMatching import match
import matplotlib.pyplot as plt

#%%
def evaluation(l1, l2):
    return (np.sum(l1 == l2)*1.0/len(l2))*100

#%%
def create_table(y_pred_d0, y_pred_d, y_test):
    return_text = ""
    return_text += ("Original, L1 measure: " + str(evaluation(y_pred_d0[:,0], y_test).round(1)) + "%,     ")
    return_text += ("Reduced, L1 measure: " + str(evaluation(y_pred_d[:,0], y_test).round(1)) + "% \n")
    return_text += ("Original, L2 measure: " + str(evaluation(y_pred_d0[:,1], y_test).round(1)) + "%,     ")
    return_text += ("Reduced, L2 measure: " + str(evaluation(y_pred_d[:,1], y_test).round(1)) + "% \n")
    return_text += ("Original, Cosine measure: " + str(evaluation(y_pred_d0[:,2], y_test).round(1)) + "%, ")
    return_text += ("Reduced, Cosine measure: " + str(evaluation(y_pred_d[:,2], y_test).round(1)) + "% \n")
    return return_text

#%%
def dimension_plot(x_train, x_test, y_train, y_test):
    dim_fv = np.arange(10, 140, 10)
    rates = np.zeros(len(dim_fv))
    for i, v in enumerate(dim_fv):
        _, d3 = match(x_train, y_train, x_test, y_test, True, n_comp=v)
        rates[i] = (evaluation(d3[:,2], y_test))
        
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel("Dimensionality of the feature vector")
    ax.set_ylabel("Correct regonition rate")
    ax.plot(np.arange(10, 140, 10), rates, label="recognition rate", marker='o')
    ax.legend(loc="best", fontsize="large")
    plt.show()

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
    
    return (FP*1.0/(TP+FP) if TP+FP != 0 else 0, 
            FN*1.0/(TN+FN) if TN+FN != 0 else 0, 
            TP*1.0/(TP+FN) if TP+FN != 0 else 0, 
            FP*1.0/(FP+TN) if TP+FN != 0 else 0)
    
#%%
def create_table2(val_d, y_pred_d):
    fm_446 = falsematch(val_d[:,2], y_pred_d[:,2], 0.446)
    fm_472 = falsematch(val_d[:,2], y_pred_d[:,2], 0.472)
    fm_502 = falsematch(val_d[:,2], y_pred_d[:,2], 0.502)

    return_text = ""
    return_text += ("False Match (Threshold: 0.446): " + str(round(fm_446[0], 3)) + ", ")
    return_text += ("False Non Match (Threshold: 0.446): " + str(round(fm_446[1], 3)) + "\n")
    return_text += ("False Match (Threshold: 0.472): " + str(round(fm_472[0], 3)) + ", ")
    return_text += ("False Non Match (Threshold: 0.472): " + str(round(fm_472[1], 3)) + "\n")
    return_text += ("False Match (Threshold: 0.502): " + str(round(fm_502[0], 3)) + ", ")
    return_text += ("False Non Match (Threshold: 0.502): " + str(round(fm_502[1], 3)) + "\n")
    return return_text
    
#%%
def fp_plot(val_d, y_pred_d):
    thresh = np.arange(0.1, 0.8, 0.002)
    false_match = []
    false_nonmatch = []
    true_positive = []
    false_positive = []
    for i in thresh:
        res = falsematch(val_d[:,2], y_pred_d[:,2], i)
        false_match.append(res[0])
        false_nonmatch.append(res[1])
        true_positive.append(res[2])
        false_positive.append(res[3])

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.set_xlabel("False Match")
    ax.set_ylabel("False Non-Match")
    ax.plot(false_match, false_nonmatch)
    
    plt.show()
