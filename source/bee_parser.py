__authors__ = 'shuffleres', 'smithc4'

import numpy as np
import cv2
import cv
from video_loader import load_local
import segmentation, features
import time
import sys

countClicks = 0

def parser(videoname):

    a,b,c = videoname.split("/")
    stfile = c.split(".")
    global storefile
    storefile = stfile[0]
    global string
    string = ""
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
    cv2.namedWindow("Current frame")
    cv2.setMouseCallback("Current frame", on_mouse)
    print("Press ESC to exit with changes.")
    print("Press Q to exit without changes.")
    print("Press the spacebar for the next frame.")
    print("Press the left mouse button on the frame to count the bees.")
    while(ret):
        timei = time.clock()*1000
        timef = time.clock()*1000
        timed = timef - timei
        framerate = framerate*.1 + .9*max(1 / 60*1000, 1 / msPerFrame*1000)
        wait = max(int(msPerFrame-timed), 1)
        key = cv2.waitKey(wait)
        if key is 113: exit()
        if key is 27: break
        if key is 32:
            countFrame = countFrame + 1

            
           
            prevFrame = currFrame.copy()
            ret, currFrame = vidcapture.read()

            if ret:
                colorFrame = currFrame.copy()
                currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
                ix,iy = -1,-1
                string += ("Frame: " + str(countFrame) + "\n")
                string += ("Clicks: " + str(countClicks) + "\n\n")
                print "Frame: " + str(countFrame) + "\n"
                global countClicks
                countClicks = 0
        cv2.imshow("Current frame", colorFrame)
    if string:
        string += ("Frame: " + str(countFrame+1) + "\n")
        string += ("Clicks: " + str(countClicks) + "\n")
        f = open("../ground_truths/" + storefile + ".txt",'w')
        print(string)
        f.write(string)
        f.close()
    

def on_mouse(event, x, y, flags, params):
    global countClicks
    global string

    if event == cv.CV_EVENT_LBUTTONDOWN:
        
        string += ("Position: " + str(x) + ', ' + str(y) + "\n")
        print(string)
        countClicks += 1;

def exit():
    cv2.destroyAllWindows()
    sys.exit()

def main():
    
    parser("../videos/729long.mkv")
    

if __name__ == '__main__':
    main()
