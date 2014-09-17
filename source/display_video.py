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
    segmentation.initBGM(frameBuffer)
    cv2.imshow("Background Model", segmentation.M_bg)

    msPerFrame = 1 / fps * 1000
    while ret:
        timei = time.clock()*1000

        dFrame = np.abs(prevFrame.astype(np.int) 
                - currFrame.astype(np.int)).astype(np.uint8)
        motionImage = ((dFrame >= 20) * 255).astype(np.uint8)
        
        bgDiff = segmentation.bgSub(currFrame)
        segmentation.updateBGM(currFrame, .05)

        kernel = np.ones((3,3), np.uint8)
        motionImage = cv2.morphologyEx(motionImage, cv2.MORPH_CLOSE, kernel, iterations=1)
        motionImage = cv2.morphologyEx(motionImage, cv2.MORPH_OPEN, kernel, iterations=1)

        if video_params['tracking']:
            contours, hierarchy = cv2.findContours(((bgDiff.copy() >= 20)*255).astype(np.uint8)
                , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            gm = features.feature_gradientMagnitude(currFrame)
            cv2.drawContours(colorFrame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)
            for contour in contours:
                # rect =  cv2.minAreaRect(contour)
                # box = cv2.cv.BoxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(colorFrame, [box], -1, (0,0,255),2)
            
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(colorFrame, (x,y), (x+w, y+h), (0,0,255),2)

        cv2.putText(colorFrame, "FPS: %d"%(framerate), (0, vres - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.CV_AA)

        # Show frames
        cv2.imshow("BG Diff", bgDiff)
        cv2.imshow("Motion", motionImage)
        cv2.imshow("Difference", dFrame)
        cv2.imshow("Current frame", colorFrame)
        
        # Copy into previous frame and get next frame
        prevFrame = currFrame.copy()
        ret, currFrame = vidcapture.read()
        if ret: 
            colorFrame = currFrame.copy()
            currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)

        # Timing 
        timef = time.clock()*1000
        timed = timef - timei
        framerate = framerate*.1 + .9*max(1 / timed*1000, 1 / msPerFrame*1000)
        wait = max(int(msPerFrame-timed), 1)
        key = cv2.waitKey(wait) 
        if key is 27: break
        if key is 32: cv2.waitKey()
        if key is 116: video_params['tracking'] ^= True
    vidcapture.release()
    cv2.destroyAllWindows()

def drawcontours(frame, contours):
    pass

def main():
    show_video("../videos/9-2_rpi9.mkv")
    show_video("../videos/729long.mkv")
    show_video("../videos/video1.mkv")
    show_video("../videos/724video.mkv")
    show_video("../videos/tonsOfFuckingBees.h264")

if __name__ == '__main__':
    main()
