# System
import numpy as np
import cv2
from time import clock

# Project
from tools import model_bg2, morph_openclose, cross
import drawing
import keys

ROI = (30, 290)
ROI_W = 500
ROI_H = 180

class App:
    def __init__(self, video_src, quiet=False, invisible=False, draw_contours=True):
        self.cam = cv2.VideoCapture(video_src)

        self.quiet = quiet
        self.invisible = invisible
        self.drawTracks = True
        self.drawContours = False
        self.drawFrameNum = False
        self.drawCenters = False

        # Learn the bg
        self.operator = cv2.BackgroundSubtractorMOG2(2000, 16, True)
        model_bg2(video_src, self.operator)

        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.arrivals = self.departures = 0

        self.extractor = cv2.DescriptorExtractor_create("SIFT")

    def run(self, as_script=True):
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

            kps, descs = self.compute(frame_gray)
            frame_root = cv2.drawKeypoints(frame, kps) 
            cv2.imshow("Root", frame_root)

            prev_gray = frame_gray
            prev_points = drawing.draw_blob_centers(fg_mask, frame, self.drawCenters)
            self.frame_idx += 1
            if not self.invisible:
                drawing.draw_rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H))
                if self.drawFrameNum:
                    drawing.draw_frame_num(frame, self.frame_idx)
                if self.drawContours:
                    drawing.draw_contours(frame, fg_mask)
                cv2.imshow('Tracking', frame)
                cv2.imshow("Mask", fg_mask)
                key = cv2.waitKey(100)
            else:
                key = cv2.waitKey(1)

            if key == keys.SPACE: key = cv2.waitKey()
            if key == keys.ESC: break
            if key == keys.Q: exit()
            

    def compute(self, image, eps=1e-7):
        sift = cv2.SIFT()
        # kps, descs = sift.detectAndCompute(image, np.ones_like(image))
        kps, descs = sift.detectAndCompute(image, None)
        if len(kps) == 0:
            return ([], None)

        # print(kps, descs.shape)
        # Hellinger Kernel
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return kps, descs


def main():
    clock()
    # video_src = "../videos/whitebg.h264"
    video_src = "../videos/newhive_noshadow3pm.h264"
    # video_src = "../videos/video1.mkv"
    # video_src = "../videos/newhive_shadow2pm.h264"

    app = App(video_src)
    app.run()
    cv2.destroyAllWindows()
    print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))
    print("{0} seconds elapsed.".format(clock()))


if __name__ == '__main__':
    main()
