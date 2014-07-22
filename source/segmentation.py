'''
Segmentation
============

Performs environment modeling for the purpose of motion segmentation
'''

import numpy as np
import cv2
from itertools import izip

# Temporal background count
C_bg = 0

def initBGM(initialFrames):
    N, h, w = initialFrames.shape
    bgModel = (np.sum(initialFrames, axis=0) / N).astype(np.uint8)

    global C_bg 
    C_bg = np.zeros((h, w))   

    return bgModel

def bgSub(currentFrame, bgModel):
    return np.abs(currentFrame - bgModel)

def adaptiveThreshold(differenceFrame):
    thresh, frame = cv2.threshold(differenceFrame.astype(np.uint8), 0, 255, 
        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh
    #return 10

def fgMask(differenceFrame):
    #adaptiveThresh = 30
    adaptiveThresh = adaptiveThreshold(differenceFrame)
    #print "Current threshold:", adaptiveThresh
    return adaptiveThresh, (differenceFrame >= adaptiveThresh).astype(np.uint8)

def updateBGM(bgModel, differenceFrame, currentFrame, adaptiveThresh, fixedThresh):
    learnRate = .1
    height, width = bgModel.shape    

    fixedThreshMask = differenceFrame <= fixedThresh
    adaptiveThreshMask = (differenceFrame < adaptiveThresh) & (differenceFrame >= fixedThresh)

    bgModel[adaptiveThreshMask] = currentFrame[adaptiveThreshMask]*learnRate \
        + bgModel[adaptiveThreshMask]*(1-learnRate)
    bgModel[fixedThreshMask] = currentFrame[fixedThreshMask]
    print bgModel.dtype


def updateBGM_alr(bgModel, differenceFrame, currentFrame, adaptiveThresh, fixedThresh):
    global C_bg
    print adaptiveThresh

    # Variables used in calculation
    lr = np.zeros(bgModel.shape)
    lr1 = np.zeros(bgModel.shape)
    lr2 = np.zeros(bgModel.shape)
    w1 = .5
    w2 = 1 - w1
    sigma1 = adaptiveThresh / 5
    sigma2 = 5
    zetaMin = np.ones(bgModel.shape)*90
    zetaMax = np.ones(bgModel.shape)*450

    # Calculate first learning rate
    mask = differenceFrame < adaptiveThresh
    numerator = differenceFrame[mask]**2 
    denominator = sigma1**2
    exponent = -.5*(numerator / denominator) 
    lr1[mask] = np.exp(exponent)

    # Calculate second learning rate
    mask = C_bg >= zetaMin
    numerator = (zetaMax[mask] - np.minimum(zetaMax[mask], C_bg[mask]))**2
    denominator = sigma2**2
    exponent = -.5*(numerator / denominator)
    lr2[mask] = np.exp(exponent)

    # Total learning rate is weighted sum of learning rates
    lr = w1*lr1 + w2*lr2 

    # Update background using learning rates
    bgModel = lr*currentFrame + (1 - lr)*bgModel

    # Update temporal count
    thresh_trash, bgMask = fgMask(differenceFrame)
    bgMask = 1 - bgMask
    C_bg += bgMask
    C_bg *= bgMask
