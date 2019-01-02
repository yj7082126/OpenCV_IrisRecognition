# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:51:06 2018

@author: user
"""
import numpy as np
import cv2

#%%
def normalize(row):
    name = row.name
    image = row['img']
    noisyimage = row['imgNoise']
    
    M = 64
    N = 512
    
    theta = np.linspace(0, 2*np.pi, N)

    diffX = row['p_posX'] - row['i_posX']
    diffY = row['p_posY'] - row['i_posY']

    a = np.ones(N) * (diffX**2 + diffY**2)
    
    if diffX < 0:
        phi = np.arctan(diffY/diffX)
        sgn = -1
    elif diffX > 0:
        phi = np.arctan(diffY/diffX)
        sgn = 1
    else:
        phi = np.pi/2
        if diffY > 0:
            sgn = 1
        else:
            sgn = -1

    b = sgn * np.cos(np.pi - phi - theta)

    r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - row['i_radius']**2))
    r = np.array([r - row['p_radius']])
    r = np.dot(np.ones([M+2,1]), r) * np.dot(np.ones([N,1]),
                        np.array([np.linspace(0,1,M+2)])).transpose()
    r = r + row['p_radius']
    r = r[1:M+1, :]

    xcosmat = np.dot(np.ones([M,1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([M,1]), np.array([np.sin(theta)]))

    x = r*xcosmat
    y = r*xsinmat

    x += row['p_posX']
    x = np.round(x).astype(int)
    x[np.where(x >= image.shape[1])] = image.shape[1] - 1
    x[np.where(x < 0)] = 0
    
    y += row['p_posY']
    y = np.round(y).astype(int)
    y[np.where(y >= image.shape[0])] = image.shape[0] - 1
    y[np.where(y < 0)] = 0
    
    newImg = image[y, x]
    noiseImg = noisyimage[y, x]
    
    coords = np.where((np.isnan(noiseImg)))
    newImg2 = newImg
    newImg2[coords] = 0.5
    avg = np.sum(newImg2) / (newImg.shape[0] * newImg.shape[1])
    newImg[coords] = avg
    
    cv2.imwrite('process/n_' + str(name.split('/')[3]), newImg)    
    
    return newImg


