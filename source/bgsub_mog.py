'''
Background subtraction using MOG

'''

import numpy as np
import cv2, os


def bgsub(vidfilename):
    video = cv2.VideoCapture(vidfilename)
    ret, frame = video.read()

    operator1 = cv2.BackgroundSubtractorMOG()
    operator2 = cv2.BackgroundSubtractorMOG2()

    # print dir(operator1)
    # print operator1.getAlgorithm
    # exit()

    while ret:
        cv2.imshow("Frame", frame)
        
        mask1 = operator1.apply(frame)
        mask2 = operator2.apply(mask1)
        mask2 = 1 - mask2

        diff = mask1.astype(np.int) - mask2.astype(np.int)
        diff = np.abs(diff).astype(np.uint8)

        cv2.imshow("Mask1", mask1)
        cv2.imshow("Mask2", mask2)
        cv2.imshow("Differences", diff)

        ret, frame = video.read()
        key = cv2.waitKey(5)
        if key == 32: break
        elif key == 27: 
            exit()


if __name__ == '__main__':
    videos = []
    for filename in os.listdir("../videos"):
        videos.append("../videos/" + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            print videofile
            bgsub(videofile)    
            