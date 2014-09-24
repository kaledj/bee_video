import numpy as np
import cv2
from video_loader import load_local
import segmentation, features
import time

video_params = dict(tracking = True)

def show_video(filename):
    vidcapture = load_local(filename)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) 
    frame_num= int(vidcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    fps = vidcapture.get(cv2.cv.CV_CAP_PROP_FPS)
    framerate = 0

    ret, prevFrame = vidcapture.read()
    prevFrame = cv2.cvtColor(prevFrame, cv2.cv.CV_BGR2GRAY)
    ret, currFrame = vidcapture.read()
    colorFrame = currFrame.copy()
    currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
    
    # Initialize bg model
    vres, hres = currFrame.shape
    frameBuffer = np.zeros((100, vres, hres))
    for i in range(100):
        ret, frame = vidcapture.read()
        frame2Buff = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        #cv2.GaussianBlur(frame2Buff, (5,5), 0, frame2Buff, 0)
        #frameBuffer[i] = cv2.equalizeHist(frame2Buff)
        frameBuffer[i] = frame2Buff
        cv2.imwrite("../samples/positive/test" + str(i) + ".png", frame)

def main():
    show_video("../videos/25-07-2014_12%3A39%3A48.h264")

if __name__ == '__main__':
    main()