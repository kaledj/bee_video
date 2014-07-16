'''
Segmentation
============

Performs environment modeling for the purpose of motion segmentation
'''

import numpy as np
import cv2

cap = cv2.VideoCapture('../videos/test714.h264')

fgbg = cv2.BackgroundSubtractorMOG2(20, 4, True)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
