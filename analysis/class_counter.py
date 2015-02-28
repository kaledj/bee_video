'''
Count the total number of positive (foreground) pixels in an image
'''

import cv2
import numpy as np


def count_posneg(img_file):
    if type(img_file) is str:
        image = cv2.imread(img_file, flags=-1)
    else:
        image = img_file
    dims = image.shape
    if len(dims) is 2:
        h, w = dims
    elif len(dims) is 3:
        h, w, d = dims
    else:
        raise Exception("Incorrect image shape")
    pixels = h * w
    positives = np.sum(np.sum(image))
    negatives = pixels - positives

    assert np.array_equal(np.unique(image), np.array([0, 1])) or \
        np.array_equal(np.unique(image), np.array([0])), np.array_repr(np.unique(image))
    return positives, negatives


if __name__ == '__main__':
    pos, neg = count_posneg('C:/Users/kaledj/Projects/SegmentationforCortina/images/whitebg/0.jpg.seg.bmp')
    print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos, neg))

    pos, neg = count_posneg(cv2.imread('C:/Users/kaledj/Projects/SegmentationforCortina/images/whitebg/0.jpg.seg.bmp'))
    print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos, neg))

