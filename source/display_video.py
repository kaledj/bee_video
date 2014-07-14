'''
Video Player
============

Displays the original source video with any additional data
such as track lines drawn on top. 
'''

import cv2
from video_loader import load_local

def show_video(filename):
	vidcapture = load_local(filename)
	nframes = int(vidcapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) # namespacing legacy openCV: cv2.cv.*
	frame_num= int(vidcapture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
	print "Frames: %s" %nframes
	fps = vidcapture.get(cv2.cv.CV_CAP_PROP_FPS)
	print "FPS value: %s" %fps

	ret, frame = vidcapture.read()
	height, width, depth = frame.shape
	while ret and cv2.waitKey(int(1/fps*100)) != 27:
	    cv2.putText(frame, "Frame: %d FPS: %d"%(frame_num, fps), 
	    	(0, height-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 
	    	lineType=cv2.CV_AA)
	    cv2.imshow("frameWindow", frame)
	    ret, frame = vidcapture.read()
	    frame_num += 1
	vidcapture.release()
	cv2.destroyWindow("frameWindow")

def main():
	#video = load_local("../videos/live_video/testmp4.mp4")
	show_video("../videos/test714MKV.mkv")

if __name__ == '__main__':
	main()