''' 
Create Samples
==============

Uses OpenCV create_samples to automatically generate training samples from
input files.
'''

import sys, os

OPENCV_BIN = "C:/lib/opencv/build/x64/vc12/bin/"
PROGRAM = "opencv_createsamples.exe"
COLLECTION_FILE = "../annotations.dat"
OUTPUT_FILE = "../samples/test.vec"
INPUT_SAMPLE = "../samples/positive/img4.png"
PARAMS = {  "-vec": OUTPUT_FILE,
            "-info": COLLECTION_FILE,
            "-img": INPUT_SAMPLE,
            "-bg": "../samples/bgfiles.txt",
            "-w": 54,
            "-h": 81,
            "-num": 1000,
            "-maxidev": 20,
            "-maxxangle": .5,
            "-maxyangle": .5,
            "-maxzangle": .8}

command = OPENCV_BIN + PROGRAM
for key in PARAMS:
    command = command + " " + key + " " + str(PARAMS[key])
os.system(command)

'''
Show Created .vec File
===========

Uses OPENCV create_samples to verify the .vec file
'''
def show_created_vec(filename):
    os.system(OPENCV_BIN + "opencv_createsamples.exe" 
        + " -vec " + filename 
        + " -w 54 -h 81 -show")

show_created_vec(OUTPUT_FILE)