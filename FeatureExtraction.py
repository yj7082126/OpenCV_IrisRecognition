# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:49:30 2018

@author: user
"""

import numpy as np
import itertools
import cv2
from scipy.signal import convolve2d

#%%
region = 40
ksize = 3

sigma1 = 3
gamma1 = 2

sigma2 = 4.5
gamma2 = 3

dim = 8
                   
#%%
def def_filter(size, sigma, theta, freq, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    xmax = int((size-1)/2)
    ymax = int((size-1)/2)
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    M0 = np.exp(-0.5 * ((x_theta/sigma_x)**2  + (y_theta/sigma_y)**2)) 
    M1 = np.cos(2 * np.pi * freq * np.sqrt((x_theta)**2 + (y_theta)**2))
    gb = (1/(2 * np.pi * sigma_x * sigma_y)) * M0 * M1
    return gb

#%%
def build_filters(ksize, sigma, freq, gamma):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = def_filter(ksize, sigma, theta, gamma/sigma, gamma)
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = convolve2d(img, kern, 'same')
        np.maximum(accum, fimg, accum)
    return accum

#%%   
def feature_vect(res1, res2, dim): 
    sizeX, sizeY = res1.shape    
    x = np.linspace(0, (sizeX/dim)-1, sizeX/dim)
    y = np.linspace(0, (sizeY/dim)-1, sizeY/dim)
    xy = np.array(list(itertools.product(x, y)))
    xy = xy.astype(int)
    
    patches1 = [res1[a[0]*dim:a[0]*dim+dim, a[1]*dim:a[1]*dim+dim] for a in xy]
    patches2 = [res2[a[0]*dim:a[0]*dim+dim, a[1]*dim:a[1]*dim+dim] for a in xy]
    
    mean1 = [np.mean(x) for x in patches1]
    sigma1 = [np.mean(np.abs(a-b)) for a, b in zip(patches1, mean1)]
    
    V = []
    for i, v in enumerate(mean1):
        V.append(v)
        V.append(sigma1[i])
        
    mean2 = [np.mean(x) for x in patches2]
    sigma2 = [np.mean(np.abs(a-b)) for a, b in zip(patches2, mean2)]
    
    for i, v in enumerate(mean2):
        V.append(v)
        V.append(sigma2[i])
        
    return V

#%%
def extract(row):
    image = row['Image']
    image = image[:region,]
    
    #filters1 = def_filter(ksize, sigma1, 0, gamma1/sigma1, gamma1)
    #filters2 = def_filter(ksize, sigma2, 0, gamma2/sigma2, gamma2)
    
    filters1 = build_filters(ksize, sigma1, gamma1/sigma1, gamma1)
    filters2 = build_filters(ksize, sigma2, gamma2/sigma2, gamma2)
    
    #res1 = convolve2d(image, filters1, 'same')
    #res2 = convolve2d(image, filters2, 'same')
    
    res1 = process(image, filters1)
    res2 = process(image, filters2)
    
    V = feature_vect(res1, res2, dim)
    return V
