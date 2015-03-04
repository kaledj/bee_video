
#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
from bgsub_mog import model_bg2, morph_openclose
import keys
from time import clock

lk_params = dict( winSize  = (30, 30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  minEigThreshold=1e-3)

ROI = (30, 290)
ROI_W = 500
ROI_H = 180
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


class App:
    def __init__(self, video_src):
        self.quiet = True
        self.invisible = False

        # Learn the bg
        self.operator = cv2.BackgroundSubtractorMOG2(2000, 32, True)
        model_bg2(video_src, self.operator)

        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.arrivals = self.departures = 0

    def run(self):
        cv2.namedWindow("Tracking")
        cv2.waitKey()
        prev_gray = None
        prev_points = None
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            fg_mask = self.operator.apply(frame, learningRate=-1)
            fg_mask = ((fg_mask == 255) * 255).astype(np.uint8)
            fg_mask = morph_openclose(fg_mask)
            # blob_centers(fg_mask, frame, True)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and prev_points is not None and len(prev_points) > 0:
                p0 = np.float32([point for point in prev_points]).reshape(-1, 1, 2)
                if p0 is not None:
                    for (x, y) in p0.reshape(-1, 2):
                        cv2.circle(frame, (x, y), radius=2, color=RED, thickness=-1)
                # p0 = np.float32(prev_points)
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                for p_i, p_f in zip(p0.reshape(-1, 2), p1.reshape(-1, 2)):
                    result = self.cross(p_i, p_f)
                    if not self.quiet:
                        if result > 0:
                            print("Arrival")
                        elif result < 0:
                            print("Departure")
                    if not self.invisible:
                        cv2.line(frame, tuple(p_i), tuple(p_f), RED)

            if len(self.tracks) > 0:
                # Find flow between last points and current points
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                for p_i, p_f in zip(p0.reshape(-1, 2), p1.reshape(-1, 2)):
                    result = self.cross(p_i, p_f)
                    if result > 0:
                        print("Arrival")
                    elif result < 0:
                        print("Departure")
                    cv2.line(frame, tuple(p_i), tuple(p_f), RED)
                # Check for good matches
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    # cv2.circle(frame, (x, y), 2, BLUE, -1)
                self.tracks = new_tracks
                cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            # Detect new points
            # if self.frame_idx % self.detect_interval == 0:
            #     mask = np.zeros_like(frame_gray)
            #     mask[:] = 255
            #     p = blob_centers(fg_mask)

            # prev_points = cv2.goodFeaturesToTrack(frame_gray, 10, .001, 40, mask=fg_mask)
            # if prev_points is not None:
            #     for (x, y) in prev_points.reshape(-1, 2):
            #         cv2.circle(frame, (x, y), radius=3, color=RED, thickness=-1)

            self.frame_idx += 1
            prev_gray = frame_gray
            prev_points = blob_centers(fg_mask)

            if not self.invisible:
                cv2.rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H), GREEN, 2)
                draw_frame_num(frame, self.frame_idx)
                cv2.imshow('Tracking', frame)
            key = cv2.waitKey(100)
            if key == keys.SPACE:
                key = cv2.waitKey()
            if key == keys.ESC:
                break
            if key == keys.Q:
                exit()
        print "Arrivals: {0} Departures: {1}".format(self.arrivals, self.departures)

    def cross(self, pt0, pt1):
        """
        Determines if the points enter or leave the rectangle.
        Returns 1 if the points enter the rect, -1 if it leaves, or 0 if neither.
        """
        x0 = pt0[0]
        y0 = pt0[1]
        x1 = pt1[0]
        y1 = pt1[1]

        p0in = ROI[0] < x0 < ROI[0] + ROI_W and ROI[1] < y0 < ROI[1] + ROI_H
        p1in = ROI[0] < x1 < ROI[0] + ROI_W and ROI[1] < y1 < ROI[1] + ROI_H

        if not p0in and p1in:
            self.arrivals += 1
            return 1
        elif p0in and not p1in:
            self.departures += 1
            return -1
        else:
            return 0


def draw_frame_num(frame, num):
    params = dict(fontFace=cv2.cv.CV_FONT_HERSHEY_COMPLEX,
                  fontScale=1, thickness=1)
    ret, baseline = cv2.getTextSize(str(num), **params)
    print ret
    cv2.putText(frame, str(num), org=(0, ret[1]), color=RED, **params)


def blob_centers(fg_mask, frame=None, drawcenters=False):
    contours, hierarchy = cv2.findContours((fg_mask.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        center = (x + (w/2), y + (h/2))
        if drawcenters and frame is not None:
            cv2.circle(frame, center, radius=2, color=RED, thickness=-1)
        centers.append(center)
    return centers


def main():
    clock()
    video_src = "../videos/whitebg.h264"
    # video_src = "../videos/video1.mkv"
    App(video_src).run()
    cv2.destroyAllWindows()
    print("{0} seconds elapsed.".format(clock()))


if __name__ == '__main__':
    main()
