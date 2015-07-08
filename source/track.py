import cv2
import numpy as np
import tools

PRESERVE_TRACKS_FOR = 4
TRACK_LEN = 10

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
        self.crossed = False

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
        drawActual = True
        drawPredicted = True

        if self.timeInvisible < PRESERVE_TRACKS_FOR:
            if drawActual:
                lenlh = len(self.locationHistory)
                if lenlh < TRACK_LEN:
                    lh = self.locationHistory
                else:
                    lh = self.locationHistory[lenlh - TRACK_LEN:]
                for i in range(len(lh)- 1):
                    cv2.line(frame, lh[i], lh[i + 1], (0, 0, 255))   
            if drawPredicted:
                lenph = len(self.predictionHistory)
                if lenph < TRACK_LEN:
                    ph = self.predictionHistory
                else:
                    ph = self.predictionHistory[lenph - TRACK_LEN:]
                for i in range(len(ph) - 1):
                    cv2.line(frame, ph[i], ph[i + 1], (0, 255, 0))        
            cv2.putText(frame, str(self.id), self.locationHistory[-1], 
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.5, color=(0, 255, 00))

    def checkCrossLastTwo(self, ROI, ROI_W, ROI_H):
        if len(self.locationHistory) >= 2 and self.timeInvisible == 0:
            pt0 = self.locationHistory[-2] 
            pt1 = self.locationHistory[-1]
            return tools.cross(ROI, ROI_W, ROI_H, pt0, pt1)
        else: 
            return 0

    def checkCross(self, ROI, ROI_W, ROI_H):
        if len(self.locationHistory) >= 2:
            pt0 = self.locationHistory[0] 
            pt1 = self.locationHistory[-1]
            return tools.cross(ROI, ROI_W, ROI_H, pt0, pt1)
        else: 
            return 0
