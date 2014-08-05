'''
Video Player
============

Displays the original source video with any additional data
such as track lines drawn on top. 
'''
import numpy as np
import cv2
from video_loader import load_local
import segmentation, features

def show_video(filename):
    vidcapture = load_local(filename)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) 
    frame_num= int(vidcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    fps = vidcapture.get(cv2.cv.CV_CAP_PROP_FPS)

    ret, prevFrame = vidcapture.read()
    prevFrame = cv2.cvtColor(prevFrame, cv2.cv.CV_BGR2GRAY)
    ret, currFrame = vidcapture.read()
    colorFrame = currFrame.copy()
    currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
    while ret:
        key = cv2.waitKey(int(1/fps*1000))
        if key is 27: break
        if key is 32: cv2.waitKey()
        dFrame = np.abs(prevFrame.astype(np.int) - currFrame.astype(np.int)).astype(np.uint8)
        motionImage = ((dFrame >= 20) * 255).astype(np.uint8)

        kernel = np.ones((3,3), np.uint8)
        #motionImage = cv2.morphologyEx(motionImage, cv2.MORPH_CLOSE, kernel, iterations=1)
        #motionImage = cv2.morphologyEx(motionImage, cv2.MORPH_OPEN, kernel, iterations=1)

        cv2.imshow("Difference", dFrame)
        cv2.imshow("Motion", motionImage)
        contours, hierarchy = cv2.findContours(motionImage, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_TC89_L1)
        cv2.imshow("GM", features.feature_gradientMagnitude(currFrame))
        prevFrame = currFrame.copy()
        cv2.drawContours(colorFrame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)
        for contour in contours:
            rect =  cv2.minAreaRect(contour)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(colorFrame, [box], -1, (0,0,255),2)
        cv2.imshow("Current frame", colorFrame)
        ret, currFrame = vidcapture.read()
        colorFrame = currFrame.copy()
        if ret: currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
    vidcapture.release()
    cv2.destroyAllWindows()

def main():
    show_video("../videos/721test.mkv")
    show_video("../videos/video1.mkv")
    show_video("../videos/724video.mkv")
    #show_video("../videos/tonsOfFuckingBees.h264")

if __name__ == '__main__':
    main()
