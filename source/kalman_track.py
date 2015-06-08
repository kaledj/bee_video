'''
Kalman Filter Based Tracking
====================
Keys
----
ESC - exit
'''

# System
import numpy as np
import cv2
import sys
from time import clock
from matplotlib import pyplot
from collections import namedtuple
from pykalman import KalmanFilter

# Project 
from tools import model_bg2, morph_openclose, cross, handle_keys
import drawing
from drawing import GREEN, RED, BLUE
import keys

ROI = (50, 200)
ROI_W = 500
ROI_H = 200

MIN_AREA = 200
MAX_AREA = 1500

observation_matrix = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0]], np.float32)

transition_matrix = np.array([[1, 0, 1, 0], 
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], np.float32)


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

        self.cam = cv2.VideoCapture(video_src)
        
        self.maxTimeInvisible = 20
        self.trackAgeThreshold = 8

        self.tracks = []
        self.frame_idx = 0
        self.arrivals = self.departures = 0

    def run(self, as_script=True):
        if self.invisible:
            cv2.namedWindow("Control")

        prev_gray = None
        prev_points = []
        nextTrackID = 0
        
        while True:
            # Get frame
            ret, frame = self.cam.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Segment
            fg_mask = self.operator.apply(frame, learningRate=-1)
            fg_mask = ((fg_mask == 255) * 255).astype(np.uint8)
            fg_mask = morph_openclose(fg_mask)
            
            # Detect blobs
            contours, _ = cv2.findContours((fg_mask.copy()), cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_TC89_L1)
            areas, detections = drawing.draw_min_ellipse(contours, frame, MIN_AREA, MAX_AREA)
            self.areas += areas

            # Track
            self.predictNewLocations()
            assignments, unmatchedTracks, unmatchedDetections = self.assignTracks(detections)
            self.updateMatchedTracks(assignments, detections)
            self.updateUnmatchedTracks(unmatchedTracks)
            self.deleteLostTracks()
            self.createNewTracks(unmatchedDetections)
            self.showTracks()

            # Store frame and go to next
            prev_gray = frame_gray
            prev_points = detections
            self.frame_idx += 1
            if not self.invisible:
                self.draw_overlays(frame, fg_mask)
                cv2.imshow('Tracking', frame)
                cv2.imshow("Mask", fg_mask)
                delay = 100
            else:
                delay = 1
            if handle_keys(delay) == 1:
                break
            
            # Should we continue running or yield some information about the current frame
            if as_script: continue
            else: pass

    def deleteLostTracks(self):
        newTracks = []
        for index, track in self.tracks:
            # Fraction of tracks age in which is was visible
            visibilty = float(track.totalVisibleCount) / track.age

            # Determine lost tracks
            if not ((track.age < self.trackAgeThreshold and visibilty < .6) or
                    (track.timeInvisible > self.maxTimeInvisible)):
                newTracks.append(track)
        self.tracks = newTracks

    def createNewTracks(self, unmatchedDetections):
        for detection in unmatchedDetections:
            # TODO: Create Kalman filter object

            # Create the new track
            newTrack = Track(nextTrackID, None)
            newTrack.locationHistory.append(detection)
            self.tracks.append(newTrack)
            self.nextTrackID += 1

    def updateMatchedTracks(self, assignments, detections):
        for assignment in assignments:
            trackIndex = assignment.trackIndex
            detectionIndex = assignment.detectionIndex
            detection = detections[detectionIndex]

            # TODO: Correct the estimate using current detection location

            # Update track
            tr = self.tracks[trackIndex]
            tr.age += 1
            tr.totalVisibleCount += 1
            tr.timeInvisible = 0
            tr.locationHistory.append(detection)

    def updateUnmatchedTracks(self, unmatchedTracks):
        for trackIndex in unmatchedTracks:
            tr = self.tracks[trackIndex]
            tr.age += 1
            tr.timeInvisible += 1

    def assignTracks(self, detections):
        Assignment = namedtuple('Assignment', 'trackIndex detectionIndex')
        assignments = []
        unmatchedTracks = []
        unmatchedDetections = []

        costMatrix = np.zeros((len(self.tracks), len(detections)))
        for i, (x1, y1) in enumerate(self.tracks):
            for j, (x2, y2) in enumerate(detections):
                costMatrix[i, j] = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
        # TODO: Implement Hungarian Assignment algorithm
        return assignments, unmatchedTracks, unmatchedDetections

    def predictNewLocations(self):
        for track in self.tracks:
            track.predict()

    def showTracks(self):
        for track in self.tracks:
            track.drawTrack()

    def draw_overlays(self, frame, fg_mask):
        drawing.draw_rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H))
        if self.drawFrameNum:
            drawing.draw_frame_num(frame, self.frame_idx)
        if self.drawContours:
            pass
            # drawing.draw_contours(frame, fg_mask)


class Track(object):
    """Represents the kalman filtered tracking history for an object"""
    def __init__(self, id, kalmanFilter):
        self.id = id
        self.kalmanFilter = kalmanFilter
        self.age = 0
        self.totalVisibleCount = 1
        self.timeInvisible = 0
        self.locationHistory = []

    def predict():
        pass

    def drawTrack(self, frame):
        cv2.polyLines(frame, [np.int32(loc) for loc in locationHistory], False, GREEN)


def main():
    clock()
    videos = []
    # video_src = "../videos/whitebg.h264"
    videos.append("../videos/newhive_noshadow3pm.h264")
    # video_src = "../videos/video1.mkv"
    videos.append("../videos/newhive_shadow2pm.h264")

    for video_src in videos:
        app = App(video_src, invisible=False, bgsub_thresh=64)
        app.run()
        cv2.destroyAllWindows()
        continue

        # Calculate area histograms
        h, w = 2, 4    
        f, axarr = pyplot.subplots(h, w)
        for i in xrange(h):
            for j in xrange(w):
                app = App(video_src, invisible=True, bgsub_thresh=2**(i*w+j+2))
                app.run()
                areas = app.areas
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
