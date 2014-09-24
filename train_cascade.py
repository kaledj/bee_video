'''
Train Cascade
=============

Uses opencv_traincascade to train a classifier. 
'''

import sys, os

OPENCV_BIN = "C:/lib/opencv/build/x64/vc12/bin/"
PROGRAM = "opencv_traincascade.exe"
INPUT_FILE = "samples/test.vec"
OUTPUT_FILE = "classifier/v2"
PARAMS = {  "-data": OUTPUT_FILE, 
            "-vec": INPUT_FILE,
            "-numStages": 20, 
            "-bg": "samples/bgfiles.txt",
            "-w": 50, 
            "-h": 58, 
            "-precalcValBufSize": 5120,
            "-precalcIdxBuffSize": 5120,
            "-numPos": 1800,
            "-numNeg": 3600,
            "featureType": "HAAR",
            "-mode": "ALL"}

command = OPENCV_BIN + PROGRAM
for key in PARAMS:
    command = command + " " + key + " " + str(PARAMS[key])
os.system(command)