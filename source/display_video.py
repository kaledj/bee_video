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

    ret, frame = vidcapture.read()
    vres, hres, channels = frame.shape

    # Initialize bg model
    frameBuffer = np.zeros((100, vres, hres))
    for i in range(100):
        ret, frame = vidcapture.read()
        frame2Buff = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        #cv2.GaussianBlur(frame2Buff, (5,5), 0, frame2Buff, 0)
        frameBuffer[i] = frame2Buff
    bgModel = segmentation.initBGM(frameBuffer)
    cv2.imshow("Background Model", bgModel)
    initialBG = bgModel.copy()

    while ret and cv2.waitKey(int(1/fps*1000)) != 27:
        frameGray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        cv2.GaussianBlur(frameGray, (5,5), 0, frameGray, 0)

        cv2.putText(frame, "Frame: %d FPS: %d"%(frame_num, fps), 
            (0, vres-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 
            lineType=cv2.CV_AA)
        
        # Get mask using current background model and filter it
        dFrame = np.abs(bgModel.astype(np.int) - frameGray.astype(np.int))
        threshold, fgmask = segmentation.fgMask(dFrame)

        kernel = np.ones((3,3), np.uint8)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Show masked frame
        #cv2.imshow("frameWindow", frame * np.dstack((fgmask, fgmask, fgmask)))
        cv2.imshow("frameWindow", frameGray * fgmask)
        cv2.imshow("Background Model", bgModel)
        cv2.imshow("Difference Image", dFrame.astype(np.uint8))
        cv2.imshow("Grayscale input", frameGray)
        # Update background model
        segmentation.updateBGM_alr(bgModel, dFrame, frameGray, threshold, 5)

        ret, frame = vidcapture.read()
        frame_num += 1
    vidcapture.release()
    cv2.destroyWindow("frameWindow")
    print initialBG-bgModel

def main():
    show_video("../videos/721test.mkv")
    show_video("../videos/live_video/test.avi")

if __name__ == '__main__':
    main()
