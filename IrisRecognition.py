# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:14:04 2018

@author: user
"""

import cv2
import numpy as np
import pandas as pd
from IrisLocalization import IrisLoc
from IrisNormalization import normalize
from ImageEnhancement import enhancement
from FeatureExtraction import extract
from IrisMatching import match
from PerformanceEvaluation import evaluation

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
print('loc')
train_local = train_image.apply(lambda x: IrisLoc(x['Image'], str(x.name)), axis=1, result_type='expand')
train_local.columns = ['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img']
test_local = test_image.apply(lambda x: IrisLoc(x['Image'], str(x.name)), axis=1, result_type='expand')
test_local.columns = ['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img']

#%%
print('norm')
train_norm = train_local.apply(lambda x: normalize(x), axis=1)
train_norm = train_norm.to_frame()
train_norm.columns = ['Image']
test_norm = test_local.apply(lambda x: normalize(x), axis=1)
test_norm = test_norm.to_frame()
test_norm.columns = ['Image']

#%%
print('enh')
train_enh = train_norm.apply(enhancement, axis=1)
train_enh = train_enh.to_frame()
train_enh.columns = ['Image']
test_enh = test_norm.apply(enhancement, axis=1)
test_enh = test_enh.to_frame()
test_enh.columns = ['Image']

#%%
print('feature')
train_feature = train_enh.apply(extract, axis=1)
#train_feature = train_feature.to_frame()
test_feature = test_enh.apply(extract, axis=1)
#test_feature = test_feature.to_frame()

#%%
x_train = np.array(train_feature.values.tolist())
y_train = np.ravel([[x, x, x] for x in range(1, 109)])
x_test = np.array(test_feature.values.tolist())
y_test = np.ravel([[x, x, x, x] for x in range(1, 109)])

#%%
y_pred_d0 = match(x_train, y_train, x_test, y_test, reduction=False)
y_pred_d = match(x_train, y_train, x_test, y_test, reduction=True, n_comp=120)

print(evaluation(y_pred_d0[:,0], y_test))
print(evaluation(y_pred_d0[:,1], y_test))
print(evaluation(y_pred_d0[:,2], y_test))
print(evaluation(y_pred_d[:,0], y_test))
print(evaluation(y_pred_d[:,1], y_test))
print(evaluation(y_pred_d[:,2], y_test))
