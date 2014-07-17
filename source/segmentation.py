'''
Segmentation
============

Performs environment modeling for the purpose of motion segmentation
'''

import numpy as np
import cv2
from itertools import izip

def initBGM(initialFrames):
    N, h, w = initialFrames.shape
    bgModel = (np.sum(initialFrames, axis=0) / N).astype(np.uint8)
    return bgModel

def bgSub(currentFrame, bgModel):
    return np.abs(currentFrame - bgModel)

def adaptiveThreshold(differenceFrame):
    thresh, frame = cv2.threshold(differenceFrame.astype(np.uint8), 0, 255, 
        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def fgMask(differenceFrame):
    #adaptiveThresh = 30
    adaptiveThresh = adaptiveThreshold(differenceFrame)
    print "Current threshold:", adaptiveThresh
    return adaptiveThresh, (differenceFrame >= adaptiveThresh)

def updateBGM(bgModel, differenceFrame, currentFrame, adaptiveThresh, fixedThresh):
    learnRate = .1
    height, width = bgModel.shape
    

    fixedThreshMask = differenceFrame <= fixedThresh
    adaptiveThreshMask = (differenceFrame < adaptiveThresh) & (differenceFrame >= fixedThresh)

    bgModel[fixedThreshMask] = currentFrame[fixedThreshMask]
    bgModel[adaptiveThreshMask] = currentFrame[adaptiveThreshMask]*learnRate \
        + bgModel[adaptiveThreshMask]*(1-learnRate)