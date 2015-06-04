#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Keys
----
ESC - exit
'''

# System
import numpy as np
import cv2
from time import clock
from matplotlib import pyplot

# Project
from tools import model_bg2, morph_openclose, cross, handle_keys
import drawing
import keys

lk_params = dict( #winSize  = (30, 30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  minEigThreshold=1e-3)

ROI = (50, 200)
ROI_W = 500
ROI_H = 200

MIN_AREA = 200
MAX_AREA = 1500

class App:
    def __init__(self, video_src, quiet=False, invisible=False, draw_contours=True, 
                 bgsub_thresh=64):
        self.quiet = quiet
        self.invisible = invisible
        self.drawTracks = True
        self.drawContours = draw_contours
        self.drawFrameNum = False

        self.areas = []

        # Learn the bg
        self.threshold = bgsub_thresh
        self.operator = cv2.BackgroundSubtractorMOG2(2000, self.threshold, True)
        model_bg2(video_src, self.operator)

        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.arrivals = self.departures = 0

    def run(self, as_script=True):
        if self.invisible:
            cv2.namedWindow("Control")
        # cv2.waitKey()

        prev_gray = None
        prev_points = None
        
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            fg_mask = self.operator.apply(frame, learningRate=-1)
            fg_mask = ((fg_mask == 255) * 255).astype(np.uint8)
            fg_mask = morph_openclose(fg_mask)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Tracking



            prev_gray = frame_gray
            contours, hier = drawing.draw_contours(frame, fg_mask)
            
            areas, prev_points = drawing.draw_min_ellipse(contours, frame, MIN_AREA, MAX_AREA)
            self.areas += areas

            self.frame_idx += 1
            if not self.invisible:
                self.draw_overlays(frame, fg_mask)
                cv2.imshow('Tracking', frame)
                cv2.imshow("Mask", fg_mask)
                delay = 33
            else:
                delay = 1
            if handle_keys(delay) == 1:
                break
            
            # Should we continue running or yield some information about the current frame
            if as_script: continue
            else: pass

        return self.areas


    def draw_overlays(self, frame, fg_mask):
        drawing.draw_rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H))
        if self.drawFrameNum:
            drawing.draw_frame_num(frame, self.frame_idx)
        if self.drawContours:
            pass
            # drawing.draw_contours(frame, fg_mask)


def main():
    clock()
    video_src = "../videos/whitebg.h264"
    # video_src = "../videos/newhive_noshadow3pm.h264"
    # video_src = "../videos/video1.mkv"
    # video_src = "../videos/newhive_shadow2pm.h264"


    app = App(video_src, invisible=False, bgsub_thresh=64)
    app.run()
    cv2.destroyAllWindows()
    exit()

    # Calculate area histograms
    h, w = 2, 4    
    f, axarr = pyplot.subplots(h, w)
    for i in xrange(h):
        for j in xrange(w):
            app = App(video_src, invisible=True, bgsub_thresh=2**(i*w+j+2))
            areas = app.run()

            axarr[i, j].set_title(
                "Threshold: {0}  Detections: {1}".format(app.threshold, len(areas)))
            axarr[i, j].hist(areas, 50, range=(0, 2000))
            
            cv2.destroyAllWindows()
            print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))
            print("{0} seconds elapsed.".format(clock()))
    pyplot.suptitle('Areas of detections in {0}'.format(video_src))
    pyplot.show()

if __name__ == '__main__':
    main()
