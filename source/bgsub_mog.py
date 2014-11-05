'''
Background subtraction using MOG

'''

import numpy as np
import cv2, os, sys
import segmentation, features, keys

def bgsub(vidfilename):
    video = cv2.VideoCapture(vidfilename)
    ret, frame = video.read()
    operator1 = cv2.BackgroundSubtractorMOG2(100,256,True)
    operator2 = cv2.BackgroundSubtractorMOG2(100,256,True)

    # print dir(operator1)
    # print operator1.getAlgorithm
    # exit()

    while ret:

        mask1 = operator1.apply(frame, learningRate = .55)
        mask2 = operator2.apply(frame, learningRate = -1)
        #mask2 = 1 - mask2

        diff = mask1.astype(np.int) - mask2.astype(np.int)
        diff = np.abs(diff).astype(np.uint8)

        cv2.imshow("Mask1", mask1)

        contours, hierarchy = cv2.findContours((mask2.copy()),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
        print(mask2)


        cv2.imshow("Mask1", mask1)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask2", mask2)
        cv2.imshow("Differences", diff)

        ret, frame = video.read()
        key = cv2.waitKey(5)
        if key == 32: break
        elif key == 27: 
            exit()


if __name__ == '__main__':
    bgsub('../videos/whitebg.h264')
    sys.exit()

    videos = []
    for filename in os.listdir("../videos"):
        videos.append("../videos/" + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            print videofile
            bgsub(videofile)    
            