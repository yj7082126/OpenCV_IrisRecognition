# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:14:04 2018

@author: user
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IrisLocalization import IrisLoc
from IrisNormalization import normalize
from ImageEnhancement import enhancement
from FeatureExtraction import extract
from IrisMatching import match
from PerformanceEvaluation import evaluation, graph, falsematch

#%% Getting file names for successive iteration
def readImg(url):
    return cv2.imread(url, 0)

trainNames = ['data/' + '{0:03d}'.format(ppl) + '/1/' + '{0:03d}'.format(ppl) + '_1_'  for ppl in list(range(1, 109))]
trainNames = [[x + str(trial) + '.bmp' for trial in list(range(1, 4))] for x in trainNames]

train_image = pd.DataFrame(columns=['Image'])
train_image['Image'] = list(map(readImg, np.ravel(trainNames)))
train_image.index = np.ravel(trainNames)

testNames = ['data/' + '{0:03d}'.format(ppl) + '/2/' + '{0:03d}'.format(ppl) + '_2_'  for ppl in list(range(1, 109))]
testNames = [[x + str(trial) + '.bmp' for trial in list(range(1, 5))] for x in testNames]

test_image = pd.DataFrame(columns=['Image'])
test_image['Image'] = list(map(readImg, np.ravel(testNames)))
test_image.index = np.ravel(testNames)

#%%
print('Starting Iris Localization')
train_local = train_image.apply(lambda x: IrisLoc(x['Image'], str(x.name)), axis=1, result_type='expand')
train_local.columns = ['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img']
test_local = test_image.apply(lambda x: IrisLoc(x['Image'], str(x.name)), axis=1, result_type='expand')
test_local.columns = ['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img']

#%%
print('Starting Iris Normalization')
train_norm = train_local.apply(lambda x: normalize(x), axis=1)
train_norm = train_norm.to_frame()
train_norm.columns = ['Image']
test_norm = test_local.apply(lambda x: normalize(x), axis=1)
test_norm = test_norm.to_frame()
test_norm.columns = ['Image']

#%%
print('Starting Iris Enhancement')
train_enh = train_norm.apply(enhancement, axis=1)
train_enh = train_enh.to_frame()
train_enh.columns = ['Image']
test_enh = test_norm.apply(enhancement, axis=1)
test_enh = test_enh.to_frame()
test_enh.columns = ['Image']

#%%
print('Starting Feature Extraction')
train_feature = train_enh.apply(extract, axis=1)
test_feature = test_enh.apply(extract, axis=1)

#%%
x_train = np.array(train_feature.values.tolist())
y_train = np.ravel([[x, x, x] for x in range(1, 109)])
x_test = np.array(test_feature.values.tolist())
y_test = np.ravel([[x, x, x, x] for x in range(1, 109)])

#%%
print('Starting Iris Matching')
val_d0, y_pred_d0 = match(x_train, y_train, x_test, y_test, reduction=False)
val_d, y_pred_d = match(x_train, y_train, x_test, y_test, reduction=True, n_comp=120)

#%%
print('Evaluation Results')
print("Original, L1 measure: " + str(evaluation(y_pred_d0[:,0], y_test).round(1)) + "%")
print("Original, L2 measure: " + str(evaluation(y_pred_d0[:,1], y_test).round(1)) + "%")
print("Original, Cosine measure: " + str(evaluation(y_pred_d0[:,2], y_test).round(1)) + "%")
print("Reduced, L1 measure: " + str(evaluation(y_pred_d[:,0], y_test).round(1)) + "%")
print("Reduced, L2 measure: " + str(evaluation(y_pred_d[:,1], y_test).round(1)) + "%")
print("Reduced, Cosine measure: " + str(evaluation(y_pred_d[:,2], y_test).round(1)) + "%")

#%%
results = graph(x_train, x_test, y_train, y_test)
fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlabel("Dimensionality of the feature vector")
ax.set_ylabel("Correct regonition rate")
ax.plot(np.arange(10, 140, 10), results, marker='o')

plt.show()

#%%
fm_446 = falsematch(val_d[:,2], y_pred_d[:,2], 0.446)
fm_472 = falsematch(val_d[:,2], y_pred_d[:,2], 0.472)
fm_502 = falsematch(val_d[:,2], y_pred_d[:,2], 0.502)
print("False Match (Threshold: 0.446): " + str(round(fm_446[0], 3)))
print("False Non Match (Threshold: 0.446): " + str(round(fm_446[1], 3)))
print("False Match (Threshold: 0.472): " + str(round(fm_472[0], 3)))
print("False Non Match (Threshold: 0.472): " + str(round(fm_472[1], 3)))
print("False Match (Threshold: 0.502): " + str(round(fm_502[0], 3)))
print("False Non Match (Threshold: 0.502): " + str(round(fm_502[1], 3)))

#%%
thresh = np.arange(0.12, 0.78, 0.02)
true_positive = []
false_positive = []
for i in thresh:
    res = falsematch(val_d[:,2], y_pred_d[:,2], i)
    true_positive.append(res[2])
    false_positive.append(res[3])
    
fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlabel("False Positive")
ax.set_ylabel("True Positive")
ax.plot(false_positive, true_positive)

plt.show()

