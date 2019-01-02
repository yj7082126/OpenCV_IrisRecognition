# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:18:18 2018

@author: user
"""

import cv2
import numpy as np

#%% Parameters

k1 = 5 # kernel size for first bilateral filter (pupil identification)
k2 = 9 # kernel size for first bilateral filter (iris identification)

window = 60 # size of window for subsetting image (120 * 120)

thresh1 = 80 # Binary image thresholding value for pupil
thresh2 = 150 # Binary image thresholding value for iris (155, 170)
scope1 = 1.6 # Lower boundary for iris radius size in respect to pupil radius
scope2 = 3.6 # Upper boundary for iris radius size in respect to iris radius
hough_list = [5,5] #hough variables for hough circles

#%% Projection: horizontal and vertical projection of image.
# Ignore the pixels in boundary of image (width: half of window (30)) for better approximation.

def projection(img):
    (h, w) = img.shape
    h = h-window
    w = w-window
    sumCols = []
    sumRows = []
    lim = int(window/2)
    for i in range(h):
        row = img[i+lim:i+lim+1, 0:w] 
        sumRows.append(np.sum(row))
    for j in range(w):
        col = img[0:h, j+lim:j+lim+1]
        sumCols.append(np.sum(col))
    return sumRows, sumCols
    
#%% Subsetting: getting a subset of image based on center point (posX, posY) and window size.
    
def subsetting(img, posX, posY, window):
    if ((posY<window) and (posX<window)):
        img = img[0:posY+window, 0:posX+window]
    elif ((posY<window) and (posX>=window)):
        img = img[0:posY+window, posX-window:posX+window]
    elif ((posY>=window) and (posX<window)):
        img = img[posY-window:posY+window, 0:posX+window]
    else:
        img = img[posY-window:posY+window, posX-window:posX+window]
    return img

#%% Thresholding: binary image thresholding & getting center of pupil from moments.
    
def thresholding(orig, posX, posY, window, otsu=True):
    img = orig.copy()
    img = subsetting(img, posX, posY, window)
    
    if otsu:
        ret,th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        ret,th = cv2.threshold(img, thresh2, 255, cv2.THRESH_BINARY_INV)
    
    M = cv2.moments(th)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    posY = int(posY+cY-window)
    posX = int(posX+cX-window)
    
    return posX, posY

#%% boundary: see if iris center is inside pupil.
# circle_detect: use houghcircles to find the iris
# circle_detectX: use houghcircles & boundary to find the pupil.
    
def boundary(x1, x2, y1, y2, r):
    if np.sqrt(np.power((y1-x1), 2) + np.power((y2-x2), 2)) < r:
        return True
    else:
        return False
    
def circle_detect(edges, dp = 20, minR = 20, maxR = 0, bound=False, posX=0, posY=0, radius=0):
    if bound == False:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, dp, 
                           param1=hough_list[0], param2=hough_list[1], 
                           minRadius = minR, maxRadius=maxR)
        if len(circles) >= 1:
            return circles[0][0] 
        else:
            return [0, 0, 0]

    else:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, dp, 
                       param1=hough_list[0], param2=hough_list[1], 
                       minRadius = minR, maxRadius=maxR)
        circles = [x for x in circles[0] if boundary(x[0], x[1], posX, posY, (radius/2))]
        if len(circles) >= 1:
            return circles[0] 
        else:
            return [posX, posY, radius]
        
def circle_detectX(edges, dp, posX, posY, radius, minR = 20, maxR = 0):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, dp, 
                       param1=hough_list[0], param2=hough_list[1], 
                       minRadius = minR, maxRadius=maxR)
    circles = [x for x in circles[0] if boundary(x[0], x[1], posX, posY, (radius/2))]
    if len(circles) >= 1:
        return circles[0]  

#%%
def IrisLoc2(orig, name): 
    (h, w) = orig.shape 
    img = orig.copy()
    
    kernel = np.ones((k1,k1),np.float32)/np.power(k1, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret,th = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(th, 0, ret)
    
    circle = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50, 
                               param1=5, param2=5, 
                               minRadius=30, maxRadius=90)[0][0]
    img_copy = orig.copy()
    
    try:
        p_posY = int(circle[1])
        p_posX = int(circle[0])
        p_radius = int(circle[2])
    except TypeError:
        print("No pupil")
        p_posY = 0
        p_posX = 0
        p_radius = 0
    
    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)   
    
    outer = np.mean([img[x] for x in list(zip(*np.where(th == 255)))])
    pupil = list(zip(*np.where(th == 0)))
    pupil = [(x[0], x[1]) for x in pupil]
        
    img = orig.copy()
    for x in pupil:
        img[x] = outer
    
    kernel = np.ones((k2,k2),np.float32)/np.power(k2, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret2,th2 = cv2.threshold(img, thresh2, 255, cv2.THRESH_BINARY)
    
    edges2 = cv2.Canny(th2, 0, ret2)
        
    circle2 = circle_detect(edges2, dp = 10, minR = 90, maxR = 120, bound=True, 
                            posX=p_posX, posY=p_posY, radius=p_radius*0.75) 
    s_posY = circle2[1]
    s_posX = circle2[0]
    s_radius = int(circle2[2])  
      
#    circle2 = circle_detectX(edges2, 10, p_posX, p_posY, p_radius*0.75, 90, 120)
#        
#    try:
#        s_posY = circle2[1]
#        s_posX = circle2[0]
#        s_radius = int(circle2[2])
#    except TypeError:
#        print("No schelra")
#        s_posY = p_posY
#        s_posX = p_posX
#        s_radius = p_radius*2
     
    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)    
    cv2.circle(img_copy,(s_posX,s_posY),s_radius,(255,255,255),2)
    cv2.circle(img_copy,(s_posX,s_posY),2,(255,255,255),3)    

    i_lowY = int(s_posY - s_radius) if int(s_posY - s_radius) >= 0 else 0
    i_highY = int(s_posY + s_radius + 1) if int(s_posY + s_radius + 1) <= h else s_radius
    
    i_lowX = int(s_posX - s_radius) if int(s_posX - s_radius) >= 0 else 0
    i_highX = int(s_posX + s_radius + 1) if int(s_posX + s_radius + 1) <= w else s_radius
    
    iris = img[i_lowY:i_highY, i_lowX:i_highX]
    
    x = np.arange(0, iris.shape[0])
    y = np.arange(0, iris.shape[1])
    
    mask = (x[np.newaxis,:]-s_radius)**2 + (y[:,np.newaxis]-s_radius)**2 > s_radius**2
    iris[mask.T] = 255
    
    topvar = int(p_posY - i_lowY - p_radius)
    topeyelid = iris[0:topvar, :]
    
    ret,th = cv2.threshold(topeyelid, 115, 255, cv2.THRESH_BINARY)
    try:
        th = cv2.Canny(th, 0, ret)
        topeyelash = th.argmax(0)
        x = np.argwhere(topeyelash > 0).reshape(1,-1)[0]
        y = topeyelash[topeyelash > 0]
    except AttributeError:
        y = []

    img_copy2 = orig.copy()
    
    if len(y) > 50:
        P = np.array([np.square(x), x, np.ones(len(x))])
        Q = np.matmul(np.linalg.inv(np.matmul(P, P.T)), np.matmul(P, y.reshape(-1,1)))
        
        x = np.arange(0, len(topeyelash), 1)
        y = (Q[0][0]*(x**2)+Q[1][0]*x+Q[2][0])
        
        for i, el in enumerate(y):
            if el > 0 and el < topeyelid.shape[0]:
                img_copy[int(el)+i_lowY, i+i_lowX] = 255
                    
        cv2.imwrite('process/l_' + str(name.split('/')[3]), img_copy)   
        
        for i, el in enumerate(y):
            if el > 0:
                img_copy2[0:int(el)+i_lowY, i+i_lowX] = np.nan
    else:
        cv2.imwrite('process/l_' + str(name.split('/')[3]), img_copy)  
        
    return [p_posX, p_posY, p_radius, s_posX, s_posY, s_radius, orig, img_copy2]

#%% 
def localize(orig, name):  
    (h, w) = orig.shape        
     
    sumRows, sumCols = projection(orig)
    
    posX = np.argmin(sumCols) + int(window/2)
    posY = np.argmin(sumRows) + int(window/2)
    
    posX, posY = thresholding(orig, posX, posY, window)
    
    posX, posY = thresholding(orig, posX, posY, window)
    
    img = orig.copy()
    img = subsetting(img, posX, posY, window)
    kernel = np.ones((k1,k1),np.float32)/np.power(k1, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret,th = cv2.threshold(img, thresh1, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(th, 0, ret)
    
    circle = circle_detect(edges, minR=20, maxR=70)
    
    img_copy = orig.copy()
    
    p_posY = int(posY+circle[1]-window)
    p_posX = int(posX+circle[0]-window)
    p_radius = int(circle[2])
    
#    try:
#        p_posY = int(posY+circle[1]-window)
#        p_posX = int(posX+circle[0]-window)
#        p_radius = int(circle[2])
#    except TypeError:
#        print("No pupil")
#        p_posY = 0
#        p_posX = 0
#        p_radius = 0
    
    if np.sqrt(np.power(p_posX-(h/2), 2) + np.power(p_posY-(w/2), 2)) >= 80:
        return IrisLoc2(orig, name)
        
    outer = np.mean([img[x] for x in list(zip(*np.where(th == 255)))])
    pupil = list(zip(*np.where(th == 0)))
    pupil = [(posY + x[0] - window, posX + x[1] - window) for x in pupil]
    
    img = orig.copy()
    for x in pupil:
        img[x] = outer

    kernel = np.ones((k2,k2),np.float32)/np.power(k2, 2)
    img2 = cv2.filter2D(img,-1,kernel)
    
    ret2,th2 = cv2.threshold(img2, thresh2, 255, cv2.THRESH_BINARY)
    
    edges2 = cv2.Canny(th2, 0, ret2)
    
    if int(name.split('/')[2]) == 1:
        circle2 = circle_detect(edges2, dp = 10, minR = 90, maxR = 140, bound=True, 
                            posX=p_posX, posY=p_posY, radius=p_radius*0.75) 
    else:
        circle2 = circle_detect(edges2, dp = 10, minR = 90, maxR = 120, bound=True, 
                            posX=p_posX, posY=p_posY, radius=p_radius*0.75) 
    
    s_posY = circle2[1]
    s_posX = circle2[0]
    s_radius = int(circle2[2]) 
    
#    try:
#        s_posY = int(circle2[1])
#        s_posX = int(circle2[0])
#        s_radius = int(circle2[2])
#    except TypeError:
#        print("No schelra")
#        return IrisLoc2(orig, name)

    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)    
    cv2.circle(img_copy,(s_posX,s_posY),s_radius,(255,255,255),2)
    cv2.circle(img_copy,(s_posX,s_posY),2,(255,255,255),3)    

    img_copy2 = orig.copy()
    
    i_lowY = int(s_posY - s_radius) if int(s_posY - s_radius) >= 0 else 0
    i_highY = int(s_posY + s_radius + 1) if int(s_posY + s_radius + 1) <= h else s_radius
    
    i_lowX = int(s_posX - s_radius) if int(s_posX - s_radius) >= 0 else 0
    i_highX = int(s_posX + s_radius + 1) if int(s_posX + s_radius + 1) <= w else s_radius
    
    iris = img[i_lowY:i_highY, i_lowX:i_highX]
    
    x = np.arange(0, iris.shape[0])
    y = np.arange(0, iris.shape[1])
    
    mask = (x[np.newaxis,:]-s_radius)**2 + (y[:,np.newaxis]-s_radius)**2 > s_radius**2
    iris[mask.T] = 255
    
    topvar = int(p_posY - i_lowY - p_radius)
    topeyelid = iris[0:topvar, :]
    
    ret,th = cv2.threshold(topeyelid, 115, 255, cv2.THRESH_BINARY)
    try:
        th = cv2.Canny(th, 0, ret)
        topeyelash = th.argmax(0)
        x = np.argwhere(topeyelash > 0).reshape(1,-1)[0]
        y = topeyelash[topeyelash > 0]
    except AttributeError:
        y = []

    if len(y) > 50:
        P = np.array([np.square(x), x, np.ones(len(x))])
        Q = np.matmul(np.linalg.inv(np.matmul(P, P.T)), np.matmul(P, y.reshape(-1,1)))
        
        x = np.arange(0, len(topeyelash), 1)
        y = (Q[0][0]*(x**2)+Q[1][0]*x+Q[2][0])
        
        for i, el in enumerate(y):
            if el > 0 and el < topeyelid.shape[0]:
                img_copy[int(el)+i_lowY, i+i_lowX] = 255
                    
        cv2.imwrite('process/l_' + str(name.split('/')[3]), img_copy)   
        
        for i, el in enumerate(y):
            if el > 0:
                img_copy2[0:int(el)+i_lowY, i+i_lowX] = np.nan
    else:
        cv2.imwrite('process/l_' + str(name.split('/')[3]), img_copy)  
        
    return [p_posX, p_posY, p_radius, s_posX, s_posY, s_radius, orig, img_copy2]