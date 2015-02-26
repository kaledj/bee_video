__authors__ = 'shuffleres', 'smithc4'

import keys
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

    ret, currFrame = vidcapture.read()
    countFrame = 0
    cv2.namedWindow("Current frame")
    cv2.setMouseCallback("Current frame", on_mouse)

    print("Press ESC to exit with changes.")
    print("Press Q to exit without changes.")
    print("Press the spacebar for the next frame.")
    print("Press the left mouse button on the frame to count the bees.")

    while(ret):
        cv2.imshow("Current frame", currFrame)
        key = cv2.waitKey()
        if key is 113:
            exit()
        if key is keys.ESC:
            break
        if key is keys.SPACE:
            countFrame += 1
            global countClicks
            print "Frame: " + str(countFrame) + "\n"
            print "Clicks: " + str(countClicks) + "\n"
            string += ("Frame: " + str(countFrame) + "\n")
            string += ("Clicks: " + str(countClicks) + "\n\n")
            countClicks = 0
            ret, currFrame = vidcapture.read()
    if string:
        string += ("Frame: " + str(countFrame+1) + "\n")
        string += ("Clicks: " + str(countClicks) + "\n")
        f = open("../ground_truths/" + storefile + ".txt",'w')
        print("|---------Information generated---------|\n")
        print(string)
        print("|---------------------------------------|\n")
        print("Storing information into ../ground_truths/" + storefile + ".txt")
        f.write(string)
        f.close()
    

def on_mouse(event, x, y, flags, params):
    global countClicks
    global string
    if event == cv.CV_EVENT_LBUTTONDOWN:
        string += ("Position: " + str(x) + ', ' + str(y) + "\n")
        countClicks += 1;

def exit():
    cv2.destroyAllWindows()
    sys.exit()

def main():
    
    parser("../videos/729long.mkv")
    

if __name__ == '__main__':
    main()
