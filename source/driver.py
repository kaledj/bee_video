'''
Top Level Driver
================

Coordinates all stages of processing pipeline. 

1) Loads video stream 
2) Preprocess video file
3) Generate good features
4) Track features 
5) Analyze flow vectors over time
'''

from video_loader import load_local
from video_writer import save_locally
from display_video import show_video

def main():
	input = "../videos/test90fps.h264"
	output = "../videos/testoutput.mkv"
	show_video(input)
	show_video(input)
	save_locally(input, output)

if __name__ == '__main__':
	main()
