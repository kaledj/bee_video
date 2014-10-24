__author__ = 'shuffleres'

import numpy as np
import cv2
import cv
from video_loader import load_local
import segmentation, features
import time

def parser(videoname, storefile):
    vidcapture = load_local(videoname)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frame_num = int(vidcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    fps = vidcapture.get(cv2.cv.CV_CAP_PROP_FPS)
    framerate = 0

    ret, prevFrame = vidcapture.read()
    prevFrame = cv2.cvtColor(prevFrame, cv2.cv.CV_BGR2GRAY)
    ret, currFrame = vidcapture.read()
    colorFrame = currFrame.copy()
    currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)


    msPerFrame = 1 / fps * 1000
    countFrame = 0
    countClicks = 0
    while(ret):
        timei = time.clock()*1000
        timef = time.clock()*1000
        timed = timef - timei
        framerate = framerate*.1 + .9*max(1 / 60*1000, 1 / msPerFrame*1000)
        wait = max(int(msPerFrame-timed), 1)
        key = cv2.waitKey(wait)

        if key is 27: break
        if key is 32:
            countFrame = countFrame + 1
            print("Frames: " + str(countFrame))
            prevFrame = currFrame.copy()
            ret, currFrame = vidcapture.read()

            if ret:
                if cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    countClicks = countClicks + 1
                    cv2.EVENT_LBUTTONUP = True
                colorFrame = currFrame.copy()
                currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
                ix,iy = -1,-1

                print("Clicks: " + str(countClicks))
                    #storefile.write("Frame: " + i + " X: " + ix + " Y: " + iy)
        cv2.imshow("Current frame", colorFrame)


def on_mouse(event, x, y, flags, params):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [x, y]
        boxes.append(sbox)
    elif event == cv.CV_EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y]
        boxes.append(ebox)


                #drawing = False # true if mouse is pressed
                #mode = True # if True, draw rectangle. Press 'm' to toggle to curve

#def on_mouse(event, x, y, flag, params):
    #global start_draw
    #global roi_x0
    #global roi_y0
    #global roi_x1
   # global roi_y1
    #if (event == cv.CV_EVENT_LBUTTONDOWN):
   # if (not start_draw):
   #     roi_x0 = x
     #   roi_y0 = y
        #start_draw = True
   # else:
      #  roi_x1 = x
      ##  roi_y1 = y
        #start_draw = False
    #elif (event == cv.CV_EVENT_MOUSEMOVE and start_draw):
        #Redraw ROI selection
       # image2 = cv.CloneImage(image)
       # cv.Rectangle(image2, (roi_x0, roi_y0), (x, y),cv.CV_RGB(255, 0, 255), 1)
        #cv.ShowImage(window_name, image2)

def main():
    parser("../videos/easy_video.h264", "test.txt")

if __name__ == '__main__':
    main()
