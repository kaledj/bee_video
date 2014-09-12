'''
Generate Negative
=================

Creates a static background for a camera to be used in 
negative sample generation.
'''

import os, sys, cv2
import numpy as np

BGR2GRAY = cv2.cv.CV_BGR2GRAY

INPUT_FILES = ["videos/video1.mkv", "videos/724video.mkv"]
OUTPUT_FILES = ["samples/negative/video1.jpg", "samples/negative/724video.jpg"]

for input, output in zip(INPUT_FILES, OUTPUT_FILES):
    # Open video
    vidcapture = cv2.VideoCapture(input)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FPS))
    ret, frame = vidcapture.read()
    frame = cv2.cvtColor(frame, BGR2GRAY)
    vres, hres = frame.shape

    # Average first 3 seconds of frames
    frameBuffer = np.zeros((fps * 3, vres, hres))
    for i in range(fps * 3):
        ret, frame = vidcapture.read()
        frame = cv2.cvtColor(frame, BGR2GRAY)
        frameBuffer[i] = frame
    N, h, w = frameBuffer.shape
    bg = (np.sum(frameBuffer, axis=0) / N).astype(np.uint8)

    # Write to file
    cv2.imwrite(output, bg)

    print input, " background written to: ", output 