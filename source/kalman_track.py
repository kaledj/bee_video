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
# from pykalman import KalmanFilter

# Project
import tools 
from tools import model_bg2, morph_openclose, cross, handle_keys
import drawing
from drawing import GREEN, RED, BLUE
import keys
from background_subtractor import BackgroundSubtractor

ROI = (50, 200)
ROI_W = 500
ROI_H = 200

MIN_AREA = 200
MAX_AREA = 1500


TRANSITION_MATRIX = np.array([[1, 0, 1, 0], 
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], np.float32)

MEASUREMENT_MATRIX = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0]], np.float32)




class App:
    def __init__(self, video_src, quiet=False, invisible=False, draw_contours=True, 
                 bgsub_thresh=64, drawTracks=True, drawFrameNum=False):
        self.quiet = quiet
        self.invisible = invisible
        self.drawContours = draw_contours
        self.threshold = bgsub_thresh
        self.drawTracks = drawTracks
        self.drawFrameNum = drawFrameNum

        self.areas = []

        # Learn the bg
        self.operator = BackgroundSubtractor(2000, self.threshold, True)
        self.operator.model_bg2(video_src)

        self.cam = cv2.VideoCapture(video_src)
        
        self.maxTimeInvisible = 8
        self.trackAgeThreshold = 4

        self.tracks = []
        self.frame_idx = 0
        self.arrivals = self.departures = 0

    def run(self, as_script=True):
        if self.invisible:
            cv2.namedWindow("Control")

        prev_gray = None
        prev_points = []
        self.nextTrackID = 0
        
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
            _, contours, _ = cv2.findContours((fg_mask.copy()), cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_TC89_L1)
            areas, detections = drawing.draw_min_ellipse(contours, frame, MIN_AREA, MAX_AREA)
            self.areas += areas

            # Track
            self.predictNewLocations(frame)
            assignments, unmatchedTracks, unmatchedDetections = self.assignTracks(detections, frame)
            self.updateMatchedTracks(assignments, detections)
            self.updateUnmatchedTracks(unmatchedTracks)
            self.deleteLostTracks()
            self.createNewTracks(detections, unmatchedDetections)
            self.showTracks(frame)

            # Store frame and go to next
            prev_gray = frame_gray
            prev_points = detections
            self.frame_idx += 1
            if not self.invisible:
                self.draw_overlays(frame, fg_mask)
                cv2.imshow('Tracking', frame)
                cv2.imshow("Mask", fg_mask)
                delay = 50
            else:
                delay = 1
            if handle_keys(delay) == 1:
                break
            
            # Should we continue running or yield some information about the current frame
            if as_script: continue
            else: pass

    def deleteLostTracks(self):
        newTracks = []
        tracksLost = 0
        for track in self.tracks:
            # Fraction of tracks age in which is was visible
            visibilty = float(track.totalVisibleCount) / track.age

            # Determine lost tracks
            if not ((track.age < self.trackAgeThreshold and visibilty < .6) or
                    (track.timeInvisible > self.maxTimeInvisible)):
                newTracks.append(track)
            else:
                tracksLost += 1
        # print("Tracks lost", tracksLost)        
        self.tracks = newTracks

    def createNewTracks(self, detections, unmatchedDetections):
        for detectionIndex in unmatchedDetections:
            detection = detections[detectionIndex]
            array_detection = np.array(detection, np.float32)
            # TODO: Create Kalman filter object
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = MEASUREMENT_MATRIX
            kf.transitionMatrix = TRANSITION_MATRIX
            # kf.processNoiseCov = PROCESS_NOISE_COV

            # Create the new track
            newTrack = Track(self.nextTrackID, kf)
            newTrack.update(array_detection)
            newTrack.locationHistory.append(detection)
            self.tracks.append(newTrack)
            self.nextTrackID += 1

    def updateMatchedTracks(self, assignments, detections):
        for assignment in assignments:
            trackIndex = assignment.trackIndex
            detectionIndex = assignment.detectionIndex
            detection = detections[detectionIndex]
            array_detection = np.array(detection, np.float32)
            track = self.tracks[trackIndex]

            # TODO: Correct the estimate using current detection location
            track.update(array_detection)

            # Update track
            track.age += 1
            track.totalVisibleCount += 1
            track.timeInvisible = 0
            track.locationHistory.append(detection)

    def updateUnmatchedTracks(self, unmatchedTracks):
        for trackIndex in unmatchedTracks:
            tr = self.tracks[trackIndex]
            tr.age += 1
            tr.timeInvisible += 1

    def assignTracks(self, detections, frame):
        """ Returns assignments, unmatchedTracks, unmatchedDetections """
        if len(self.tracks) == 0:
            # There are no tracks, all detections are unmatched
            unmatchedDetections = range(len(detections))
            return [], [], unmatchedDetections
        elif len(detections) == 0:
            # There are no detections, all tracks are unmatched
            unmatchedTracks = range(len(self.tracks))
            return [], unmatchedTracks, []
        else:
            costMatrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                x1, y1 = track.getPredictedXY()
                for j, (x2, y2) in enumerate(detections):
                    # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0))
                    costMatrix[i, j] = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
            return tools.assignment(costMatrix)

    def predictNewLocations(self, frame):
        for track in self.tracks:
            track.predict(frame)

    def showTracks(self, frame):
        if self.drawTracks:
            for track in self.tracks:
                track.drawTrack(frame)

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
        self.predictionHistory = []
        self.numCorrections = 0

    def getPredictedXY(self):
        x = np.int32(self.prediction[0])
        y = np.int32(self.prediction[1])
        return x, y

    def predict(self, frame):
        pred = self.kalmanFilter.predict()
        if self.numCorrections < 3:
            pred = self.locationHistory[-1]
        self.prediction = pred
        x = np.int32(pred[0])
        y = np.int32(pred[1])
        # cv2.putText(frame, str(self.id), (x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
        #     fontScale=.5, color=(0, 255, 0))
        self.predictionHistory.append((x, y))
        return x, y

    def update(self, detection):
        self.kalmanFilter.correct(detection)
        self.numCorrections += 1

    def drawTrack(self, frame):
        if self.timeInvisible < 4:
            lh = self.locationHistory
            ph = self.predictionHistory
            for i in range(len(lh)- 1):
                cv2.line(frame, lh[i], lh[i + 1], (0, 0, 255))   
            for i in range(len(ph) - 1):
                cv2.line(frame, ph[i], ph[i + 1], (0, 255, 0))        
            cv2.putText(frame, str(self.id), self.locationHistory[-1], 
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.5, color=(0, 0, 255))


def main():
    print("OpenCV version: {0}".format(cv2.__version__))
    clock()
    videos = []
    # videos.append("../videos/whitebg.h264")
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

        