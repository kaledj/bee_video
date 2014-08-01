'''
Video Player
============

Displays the original source video with any additional data
such as track lines drawn on top. 
'''
import numpy as np
import cv2
from video_loader import load_local
import segmentation

def show_video(filename):
    vidcapture = load_local(filename)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) 
    frame_num= int(vidcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    fps = vidcapture.get(cv2.cv.CV_CAP_PROP_FPS)

    ret, prevFrame = vidcapture.read()
    prevFrame = cv2.cvtColor(prevFrame, cv2.cv.CV_BGR2GRAY)

    while ret and cv2.waitKey(int(1/fps*1000)) != 27:
        ret, currFrame = vidcapture.read()
        currFrame = cv2.cvtColor(currFrame, cv2.cv.CV_BGR2GRAY)
        dFrame = np.abs(prevFrame.astype(np.int) - currFrame.astype(np.int)).astype(np.uint8)
        cv2.imshow("Previous frame", prevFrame)
        prevFrame = currFrame.copy()
        motionImage = ((dFrame >= 20) * 255).astype(np.uint8)
        
        cv2.imshow("Difference", dFrame)
        cv2.imshow("Motion", motionImage)

        contours, hierarchy = cv2.findContours(motionImage, cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(currFrame, contours, -1, (0, 0, 255), hierarchy=hierarchy, maxLevel=2)
        cv2.imshow("Current frame", currFrame)
    vidcapture.release()
    cv2.destroyAllWindows()

def main():
    show_video("../videos/721test.mkv")
    show_video("../videos/video1.mkv")
    show_video("../videos/724video.mkv")

if __name__ == '__main__':
    main()
