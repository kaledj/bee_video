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
# Persistant background model
M_bg = 0

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
    #return 20

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

def updateBGM_alr(bgModel, differenceFrame, currentFrame, adaptiveThresh, fgmask):
    global C_bg, M_bg

    # Variables used in calculation
    lr = np.zeros(bgModel.shape)
    lr1 = np.zeros(bgModel.shape)
    lr2 = np.zeros(bgModel.shape)
    w1 = .5
    w2 = 1 - w1
    sigma1 = adaptiveThresh / 5
    sigma2 = 15
    zetaMin = 60
    zetaMax = 200

    # Calculate first learning rate
    mask = differenceFrame < adaptiveThresh
    numerator = differenceFrame[mask]**2 
    denominator = sigma1**2
    exponent = -.5*(numerator / denominator) 
    lr1[mask] = np.exp(exponent)

    # Calculate second learning rate
    mask = C_bg >= zetaMin
    numerator = (np.minimum(zetaMax, C_bg[mask]) - zetaMax)**2
    denominator = sigma2**2
    exponent = -.5*(numerator / denominator)
    lr2[mask] = np.exp(exponent)

    # Total learning rate is weighted sum of learning rates
    lr[:] = w1*lr1 + w2*lr2 

    # Update background using learning rates
    bgModel[:] = lr*currentFrame + (1 - lr)*bgModel
    print bgModel.dtype

    # Update temporal count
    bgMask = 1 - fgmask
    C_bg[:] = bgMask + C_bg
    C_bg[:] = bgMask * C_bg

    cv2.imshow("Counts", C_bg.astype(np.uint8))
    cv2.imshow("LR 1", (255 * lr1).astype(np.uint8))
    cv2.imshow("LR 2", (255 * lr2).astype(np.uint8))
    

    #print lr
    #print C_bg
