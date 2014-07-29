'''
Mixture of Gaussian Classsifiers
============

Uses Mixture of Gaussian model to model background
'''

import cv2
import numpy as np
from sklearn import mixture

import features

classifiers = 0

def initClassifiers(initialFrames):
    # print initialFrames.dtype
    global classifiers
    classifiers = []
    # frameFeatureVectors = []
    # N, h, w, d = initialFrames.shape
    # np.random.seed(1)
    # for i in range(13):
    #     classifiers.append(mixture.GMM())
    #     frameFeatureVectors.append(np.zeros((N, h*w)))
    # for index, frame in enumerate(initialFrames):
    #     frame = frame.astype(np.uint8)
    #     b, g, r = features.feature_bgr(frame)
    #     frameFeatureVectors[0][index] = b.flatten()
    #     frameFeatureVectors[1][index] = g.flatten()
    #     frameFeatureVectors[2][index] = r.flatten()
    #     frameFeatureVectors[3][index] = features.feature_gradientOrientation(frame).flatten()
    #     frameFeatureVectors[4][index] = features.feature_gradientMagnitude(frame).flatten()
    #     frameFeatureVectors[5][index] = features.feature_haar1(frame).flatten()
    #     frameFeatureVectors[6][index] = features.feature_haar2(frame).flatten()
    #     frameFeatureVectors[7][index] = features.feature_haar3(frame).flatten()
    #     frameFeatureVectors[8][index] = features.feature_haar4(frame).flatten()
    #     frameFeatureVectors[9][index] = features.feature_haar5(frame).flatten()
    #     frameFeatureVectors[10][index]= features.feature_haar6(frame).flatten()
    #     frameFeatureVectors[11][index] = features.feature_haar7(frame).flatten()
    #     frameFeatureVectors[12][index] = features.feature_haar8(frame).flatten()
    # for index, classifier in enumerate(classifiers):
    #     classifiers[index].fit(frameFeatureVectors[index])
    for i in range(13):
        classifiers.append(cv2.BackgroundSubtractorMOG())

def applyAll(frame):
    votes = np.zeros((480, 640))
    b, g, r = features.feature_bgr(frame)
    votes += classifiers[0].apply(b)
    votes += classifiers[1].apply(g)
    votes += classifiers[2].apply(r)
    #votes += classifiers[3].apply(features.feature_gradientOrientation(frame))
    #votes += classifiers[4].apply(features.feature_gradientMagnitude(frame))
    votes += classifiers[5].apply(features.feature_haar1(frame).astype(np.uint8))
    votes += classifiers[6].apply(features.feature_haar2(frame).astype(np.uint8))
    votes += classifiers[7].apply(features.feature_haar3(frame).astype(np.uint8))
    votes += classifiers[8].apply(features.feature_haar4(frame).astype(np.uint8))
    votes += classifiers[9].apply(features.feature_haar5(frame).astype(np.uint8))
    votes += classifiers[10].apply(features.feature_haar6(frame).astype(np.uint8))
    votes += classifiers[11].apply(features.feature_haar7(frame).astype(np.uint8))
    votes += classifiers[12].apply(features.feature_haar8(frame).astype(np.uint8))

    return votes > (len(classifiers) / 2)
