''' 
Create Samples
==============

Uses OpenCV create_samples to automatically generate training samples from
input files.
'''

import sys, os

OPENCV_BIN = "C:/lib/opencv/build/x64/vc12/bin/"

INPUT_FILES = ["samples/img1.png"]
OUTPUT_FILE = "test.vec"

for input in INPUT_FILES:
    print os.getcwd();
    os.system(OPENCV_BIN + "opencv_createsamples.exe" + " -vec " + OUTPUT_FILE 
        + " -img " + input)    


'''
Show Created .vec File
===========

Uses OPENCV create_samples to verify the .vec file
'''
def show_created_vec(filename):
    os.system(OPENCV_BIN + "opencv_createsamples.exe" + " -vec " + filename 
        + " -w 24 -h 24 -show")    

show_created_vec(OUTPUT_FILE)