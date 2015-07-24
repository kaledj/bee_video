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
from background_subtractor import BackgroundSubtractor


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  minEigThreshold=1e-3)

ROI = (100, 150)
ROI_W = 370
ROI_H = 250

MIN_AREA = 200
MAX_AREA = 1500

class App:
    def __init__(self, video_src, quiet=True, invisible=False, draw_contours=True, 
                 bgsub_thresh=64):
        self.quiet = quiet
        self.invisible = invisible
        self.drawContours = draw_contours
        self.threshold = bgsub_thresh
        self.drawTracks = True
        self.drawFrameNum = False

        self.areas = []

        # Learn the bg
        self.operator = BackgroundSubtractor(2000, self.threshold, True)
        self.operator.model_bg2(video_src)

        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.arrivals = self.departures = 0

    def run(self, as_script=True):
        if self.invisible:
            cv2.namedWindow("Control")

        prev_gray = None
        prev_points = None
        
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break

            fg_mask = self.operator.apply(frame)
            fg_mask = ((fg_mask == 255) * 255).astype(np.uint8)
            fg_mask = morph_openclose(fg_mask)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and prev_points is not None:
                p0 = np.float32([point for point in prev_points]).reshape(-1, 1, 2)
                if drawing.draw_prev_points(frame, prev_points):
                    # p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                    frame_gray[fg_mask == 0] = 255
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                    for p_i, p_f in zip(p0.reshape(-1, 2), p1.reshape(-1, 2)):
                        result = cross(ROI, ROI_W, ROI_H, p_i, p_f)
                        if result is 1:
                            self.arrivals += 1
                            if not self.quiet:
                                print("Arrival")
                        elif result is -1:
                            self.departures += 1
                            if not self.quiet:
                                print("Departure")

                        if self.drawTracks:
                            drawing.draw_line(frame, tuple(p_i), tuple(p_f))

            prev_gray = frame_gray
            contours, hier = drawing.draw_contours(frame, fg_mask)
            
            areas, prev_points = drawing.draw_min_ellipse(contours, frame, MIN_AREA, MAX_AREA)
            self.areas += areas

            self.frame_idx += 1
            if not self.invisible:
                self.draw_overlays(frame, fg_mask)
                cv2.imshow("Fas", frame_gray)
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
    print(__name__)
    clock()
    videos = []
    # video_src = "../videos/whitebg.h264"
    videos.append("../videos/newhive_noshadow3pm.h264")
    # video_src = "../videos/video1.mkv"
    videos.append("../videos/newhive_shadow2pm.h264")

    for video_src in videos:
        app = App(video_src, bgsub_thresh=64)
        app.run()
        cv2.destroyAllWindows()
        print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))
        


if __name__ == '__main__':
    main()
