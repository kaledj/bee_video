
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

lk_params = dict( winSize  = (21, 21),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ROI = (30, 290)
ROI_W = 500
ROI_H = 180
GREEN = (0, 255, 0)


class App:
    def __init__(self, video_src):
        # Learn the bg
        self.operator = cv2.BackgroundSubtractorMOG2(2000, 32, True)
        model_bg2(video_src, self.operator)

        self.track_len = 10
        self.detect_interval = 2
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            ret, frame = self.cam.read()
            cv2.rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H), GREEN, 2)
            if not ret:
                break
            fg_mask = self.operator.apply(frame, learningRate=-1)
            fg_mask = ((fg_mask == 255) * 255).astype(np.uint8)
            fg_mask = morph_openclose(fg_mask)

            blob_centers(fg_mask, frame, True)

            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame
            cv2.imshow("Illumination normalized", frame_gray)

            if len(self.tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
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
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = blob_centers(fg_mask)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            prev_gray = frame_gray

            cv2.imshow('Tracking', frame)

            key = cv2.waitKey(50)
            if key == keys.ESC:
                break
            if key == keys.SPACE:
                cv2.waitKey()
            if key == keys.Q:
                exit()


def blob_centers(fg_mask, frame=None, drawcenters=False):
    contours, hierarchy = cv2.findContours((fg_mask.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        center = (x + (w/2), y + (h/2))
        if drawcenters and frame is not None:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
        centers.append(center)
    return centers


def cross(pt0, pt1):
    '''
    Determines if the points enter or leave the rectangle.
    Returns 1 if the points enter the rect, -1 if it leaves, or 0 if neither.
    '''
    x0 = pt0[0]
    y0 = pt0[1]
    x1 = pt1[0]
    y1 = pt1[1]

    p0in = ROI[0] > x0 < ROI[0] + ROI_W and ROI[1] > y0 < ROI[1] + ROI_H
    p1in = ROI[0] > x1 < ROI[0] + ROI_W and ROI[1] > y1 < ROI[1] + ROI_H

    if not p0in and p1in:
        return 1
    elif p0in and not p1in:
        return -1
    else:
        return 0


def main():
    video_src = "../videos/whitebg.h264"
    # video_src = "../videos/video1.mkv"
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
