# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:51:06 2018

@author: user
"""
import numpy as np
import cv2

def normalize(row):

    image = row['img']
    M = 64
    N = 512
    
    theta = np.linspace(0, 2*np.pi, N)

    ox = row['p_posX'] - row['i_posX']
    oy = row['p_posY'] - row['i_posY']

    a = np.ones(N) * (ox**2 + oy**2)
    
    if ox < 0:
        phi = np.arctan(oy/ox)
        sgn = -1
    elif ox > 0:
        phi = np.arctan(oy/ox)
        sgn = 1
    else:
        phi = np.pi/2
        if oy > 0:
            sgn = 1
        else:
            sgn = -1

    b = sgn * np.cos(np.pi - phi - theta)

    r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - row['i_radius']**2))
    r = np.array([r - row['p_radius']])

    rmat = np.dot(np.ones([M+2,1]), r)

    rmat = rmat * np.dot(np.ones([N,1]),
                        np.array([np.linspace(0,1,M+2)])).transpose()
    rmat = rmat + row['p_radius']
    rmat = rmat[1 : M+1, :]

    xcosmat = np.dot(np.ones([M,1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([M,1]), np.array([np.sin(theta)]))

    xo = rmat * xcosmat
    yo = rmat * xsinmat

    xo = row['p_posX'] + xo
    xo = np.round(xo).astype(int)
    xo[np.where(xo >= image.shape[1])] = image.shape[1] - 1
    xo[np.where(xo < 0)] = 0
    
    yo = row['p_posY'] - yo
    yo = np.round(yo).astype(int)
    yo[np.where(yo >= image.shape[0])] = image.shape[0] - 1
    yo[np.where(yo < 0)] = 0
    
    polar_array = image[yo, xo]
    
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg
    
    return polar_array

#def normalize(row):
#    M = 64
#    N = 512
#    dim = (320, 280)
#    img = row['img']
#    res = np.zeros((M, N))  
#    pY = row['p_posY']
#    pX = row['p_posX']
#    pR = row['p_radius']
#    iY = row['i_posY']
#    iX = row['i_posX']
#    iR = row['i_radius']
#    
#    X = np.linspace(0, 2*np.pi, N, endpoint=False)
#    Y = np.linspace(0, 1, M, endpoint=False)
#    
#    yp = pY + pR*np.sin(X)
#    xp = pX + pR*np.cos(X)
#    
#    # get the outer boundary coordinate
#    yi = iY + iR*np.sin(X)
#    xi = iX + iR*np.cos(X)
#    
#    x = np.dot(Y.reshape(-1,1), (xi - xp).reshape(1,-1))
#    x = np.minimum(np.repeat(xp.reshape(1,-1), 64, axis=0) + x, dim[0]-1)
#    
#    y = np.dot(Y.reshape(-1,1), (yi - yp).reshape(1,-1))
#    y = np.minimum(np.repeat(yp.reshape(1,-1), 64, axis=0) + y, dim[1]-1)
#
#    for a in range(M):
#        for b in range(N):
#            res[a][b] = img[int(y[a,b])][int(x[a,b])]
#            
#    res = res.astype(np.uint8)
#    cv2.imwrite('normalize/s_' + str(row.name.split('/')[3]), res)
#    return res

