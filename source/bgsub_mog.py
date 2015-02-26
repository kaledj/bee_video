'''
Background subtraction using MOG

'''

import scipy.stats
import numpy as np
import cv2, os, sys
import segmentation, features, keys
from analysis import class_counter

BGR2GRAY = cv2.cv.CV_BGR2GRAY

def bgsub(vidfilename):
    # Start by getting a model of the background
    bg = model_bg(vidfilename)
    cv2.imshow("BG Model", bg)
    operator = cv2.BackgroundSubtractorMOG2(2000, 16, True)
    # Learn the bg
    operator.apply(bg, learningRate=1)

    color = np.random.randint(0, 255, (100, 3))

    video = cv2.VideoCapture(vidfilename)
    ret, frame = video.read()
    while ret:
        mask = operator.apply(frame, learningRate=-1)
        mask = morph_openclose(mask)
        mask_binary = mask == 255
        pos, neg = class_counter.count_posneg(mask_binary)
        print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos, neg))

        mask = ((mask == 255) * 255).astype(np.uint8)
        cv2.imshow("Mask", mask)

        contours, hierarchy = cv2.findContours((mask.copy()),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

        ret, frame = video.read()
        key = cv2.waitKey(5)
        if key == keys.ESC:
            break
        if key == keys.SPACE:
            cv2.waitKey()
        if key == keys.Q: 
            exit()


def model_bg(video):
    vidcapture = cv2.VideoCapture(video)
    ret, frame = vidcapture.read()
    frame = cv2.cvtColor(frame, BGR2GRAY)
    vres, hres = frame.shape

    # Average first N frames
    N = 100
    frameBuffer = np.zeros((N, vres, hres), np.uint32)
    for i in range(N):
        ret, frame = vidcapture.read()
        if ret:
            frame = cv2.cvtColor(frame, BGR2GRAY)
            frameBuffer[i] = frame
        else:
            break
    vidcapture.release()
    return (np.sum(frameBuffer, axis=0) / N).astype(np.uint8)


def morph_openclose(image):
    kernel = np.ones((3, 3), np.uint8)
    new_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel, iterations=1)


if __name__ == '__main__':
    # bgsub('../videos/whitebg.h264')
    # sys.exit()

    videos = []
    for filename in os.listdir("../videos"):
        videos.append("../videos/" + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            print videofile
            bgsub(videofile)    
