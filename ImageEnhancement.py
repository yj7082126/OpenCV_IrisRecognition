# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:35:42 2018

@author: user
"""

from skimage.measure import block_reduce
import numpy as np
import cv2

#%%
def enhancement(row):
    img = row['Image']
    dim = img.shape
    
    stride = 16
#    
    initialize_img = np.zeros((int(dim[0]/stride), int(dim[1]/stride)))
    
    for i in range(0,dim[0]-15,stride):
        for j in range(0,dim[1]-15,stride):
            block = img[i:i+stride, j:j+stride]
            m = np.mean(block,dtype=np.float32)
            initialize_img[i//16, j//16] = m
            
    image_set = cv2.resize(initialize_img, (dim[1],dim[0]), interpolation=cv2.INTER_CUBIC)

    enhance = img - image_set
    enhance = enhance - np.amin(enhance.ravel())
    img = enhance.astype(np.uint8)      
         
    img2 = np.zeros(dim)
    for i in range(0,img.shape[0],stride*2):
        for j in range(0,img.shape[1],stride*2):
            img2[i:i+stride*2, j:j+stride*2] = cv2.equalizeHist(img[i:i+stride*2, j:j+stride*2].astype(np.uint8))
    cv2.imwrite('enhance/s_' + str(row.name.split('/')[3]), img2)

    return img2

