''' 
Video Loader
============

Loads a video stream into memory and provides and object reference
to it. 
'''

import cv2
import sys

def load_local(filepath):
    try:
        vidfile = cv2.VideoCapture(filepath)
    except:
        print "Error: Opening input stream"
        sys.exit(1)
    if not vidfile.isOpened():
        print "Error: Opening input stream"
    return vidfile
