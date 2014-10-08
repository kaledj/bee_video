'''
Detection
=========

Implements multiple different types of detection. HAAR, SIFT...
'''

# Local modules
from video_loader import load_local
# System modules
import cv2, os, sys
import numpy as np

# ROI = {"tlcorner": (50, 50), "brcorner":(150, 150)}
ROI = (230, 230)

def cascadeDetect():
    cascade = cv2.CascadeClassifier("../classifier/v1/cascade.xml")

    videos = []
    for filename in os.listdir("../videos"):
        split = filename.split(".")
        if len(split) > 1:
            videos.append("../videos/" + filename)

    for videofile in videos:
        print videofile
        video = load_local(videofile)
        ret, frame = video.read()
        while ret:
            # frameGray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
            bees = cascade.detectMultiScale(frame, minNeighbors=2)
            for (x, y, w, h) in bees:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                center = (x+(w/2), y+(h/2))

            cv2.rectangle(frame, ROI, (ROI[0]+400, ROI[1]+230), (0, 255, 0), 2)

            cv2.imshow("Video", frame)

            ret, frame = video.read()

            key = cv2.waitKey(5)
            if key == 32: break
            elif key == 27: 
                exit()

def siftDetect():
    videos = []
    for filename in os.listdir("../videos"):
        videos.append("../videos/" + filename)

    for videofile in videos:
        video = load_local(videofile)
        ret, frame = video.read()
        while ret:
            frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
            
            sift = cv2.SIFT()
            kp = sift.detect(frame, None)
            frame = cv2.drawKeypoints(frame, kp)
            cv2.imshow("Keypoints", frame)

            ret, frame = video.read()
            key = cv2.waitKey(10)
            if key == 32: break
            elif key == 27: 
                exit()

def cross(pt0, pt1, rect):
    pt0Bool = pt0[0]>=rect.corner[0] and pt0[0] <= rect.corner[0]+rect.w and pt0[1]>=rect.corner[1] and pt0[1]<=rect.corner[1]+rect.h
    pt1Bool = pt1[0]>=rect.corner[0] and pt1[0] <= rect.corner[0]+rect.w and pt1[1]>=rect.corner[1] and pt1[1]<=rect.corner[1]+rect.h
    if not pt0Bool and pt1Bool:
        return 1
    elif pt0Bool and not pt1Bool:
        return -1
    else:
        return 0

def exit():
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    cascadeDetect()