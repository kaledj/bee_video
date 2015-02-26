'''
Count the total number of positive (foreground) pixels in an image
'''

import cv2
import numpy as np


def count_posneg(img_file):
    if type(img_file) is str:
        image = cv2.imread(img_file)
    else:
        image = img_file
    print(image.shape)
    h, w = image.shape
    pixels = h * w
    positives = np.sum(np.sum(image))
    negatives = pixels - positives

    return positives, negatives


if __name__ == '__main__':
    pos, neg = count_posneg('C:/Users/kaledj/Projects/SegmentationforCortina/images/whitebg/0.jpg.seg.bmp')
    print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos, neg))

    pos, neg = count_posneg(cv2.imread('C:/Users/kaledj/Projects/SegmentationforCortina/images/whitebg/0.jpg.seg.bmp'))
    print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos, neg))