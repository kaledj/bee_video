
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
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


class App:
    def __init__(self, video_src):
        # Start by getting a model of the background
        from bgsub_mog import model_bg
        self.bg = model_bg(video_src)

        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        cv2.imshow("BG Model", self.bg)
        operator = cv2.BackgroundSubtractorMOG2(2000, 16, True)
        # Learn the bg
        operator.apply(self.bg, learningRate=1)

        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_gray = operator.apply(frame_gray, learningRate=-1)
            frame_gray = ((frame_gray == 255) * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_CLOSE, kernel, iterations=1)
            frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, kernel, iterations=1)

            visible = frame_gray.copy()

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
                    cv2.circle(visible, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(visible, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            prev_gray = frame_gray

            cv2.imshow('lk_track', visible)
            ch = 0xFF & cv2.waitKey(10)
            if ch == 27:
                break

def main():
    video_src = "../videos/whitebg.h264"
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
