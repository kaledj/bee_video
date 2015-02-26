'''
Train Cascade
=============

Uses opencv_traincascade to train a classifier. 
'''

import sys, os

OPENCV_BIN = "C:/lib/opencv/build/x64/vc12/bin/"
PROGRAM = "opencv_traincascade.exe"
INPUT_FILE = "samples/test.vec"
OUTPUT_CASCADE_DIR = "classifier/v2verticaldown"
PARAMS = {  "-data": OUTPUT_CASCADE_DIR, 
            "-vec": INPUT_FILE,
            "-bg": "samples/bgfiles.txt",
            "-w": 54, 
            "-h": 81, 
            "-precalcValBufSize": 7168,
            "-precalcIdxBuffSize": 3072,
            "-numPos": 900,
            "-numNeg": 1800,
            "featureType": "HAAR",
            "-mode": "ALL"}

command = OPENCV_BIN + PROGRAM
for key in PARAMS:
    command = command + " " + key + " " + str(PARAMS[key])
print("Executing {0}".format(command))
os.system(command)
