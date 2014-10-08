'''
Generate Negatives
==================

Creates a static background for a camera to be used in 
negative sample generation.
'''

import os, sys, cv2, time
import numpy as np

BGR2GRAY = cv2.cv.CV_BGR2GRAY

INPUT_FILES = []
OUTPUT_FILES = []

bgfiles = open("samples/bgfiles.txt", 'w')
for filename in os.listdir("videos"):
    split = filename.split(".")
    if len(split) > 1:
        INPUT_FILES.append("videos/" + filename)
        OUTPUT_FILES.append("samples/negative/" + split[0] + ".jpg")
        bgfiles.write("negative/" + split[0] + ".jpg\n")
bgfiles.close()

count = 0
ti = time.time()
for input, output in zip(INPUT_FILES, OUTPUT_FILES):
    print "IN: ", input
    print "OUT: ", output

    # Open video
    vidcapture = cv2.VideoCapture(input)
    nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FPS))
    ret, frame = vidcapture.read()
    frame = cv2.cvtColor(frame, BGR2GRAY)
    vres, hres = frame.shape

    # Average first N frames
    N = 500
    frameBuffer = np.zeros((N, vres, hres), np.uint32)
    for i in range(N):
        ret, frame = vidcapture.read()
        if ret:
            frame = cv2.cvtColor(frame, BGR2GRAY)
            frameBuffer[i] = frame
    bg = (np.sum(frameBuffer, axis=0) / N).astype(np.uint8)

    # Write to file
    if cv2.imwrite(output, bg):
        print input, " background written to: ", output 
    else: print "Error writing ", input, " to", output
    count = count + 1
tf = time.time()
print "{0} negatives written in {1} seconds.".format(count, tf - ti)
