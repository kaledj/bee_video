''' 
Video Writer
============

Saves a processed video stream so it can be viewed without 
any recomputation. 
'''

import cv2
from video_loader import load_local

def save_locally(inputfilename, outfilename):
	# Set up input stream and get info
	vidcapture = load_local(inputfilename)
	print vidcapture.isOpened()
	print vidcapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0.0)
	print vidcapture.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
	print vidcapture.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	print vidcapture.get(0xFF)

	fps = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FPS))
	width = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

	print fps, width, height

	# Set up output stream 
	#fourcc = cv2.cv.CV_FOURCC('X','2','6','4')
	fourcc = cv2.cv.CV_FOURCC('F','M','P','4')
	output = cv2.VideoWriter(outfilename, fourcc, fps, (width, height))
	
	# Write each frame to file
	ret, frame = vidcapture.read()
	while ret:
		ret, frame = vidcapture.read()
		output.write(frame)	
	vidcapture.release()
	output.release()
