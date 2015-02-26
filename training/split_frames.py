'''
Split Frames
===============

Takes every 10th frame from a video sequence to be used in cropping out samples. 
'''

import cv2
import os


def generate_frames(filename):
    vidcapture = cv2.VideoCapture(filename)
    filename = os.path.basename(filename)
    for i in range(500):
        ret, frame = vidcapture.read()
        if ret and (i % 10 == 0):
            print cv2.imwrite("C:/Users/kaledj/Projects/SegmentationforCortina/images/whitebg/{0}.jpg".format(i), frame)
        if not ret: 
            break

def main():
    generate_frames("../videos/whitebg.h264")
    # generate_frames("videos/9-2_rpi9.mkv", 'b')
    # generate_frames("videos/724video.mkv", 'c')
    # generate_frames("videos/video1.mkv", 'd')

if __name__ == '__main__':
    main()
    exit()

    videos = []
    for filename in os.listdir("videos"):
        videos.append("videos/" + filename)
    for videofile in videos:
        print videofile
        if os.path.isfile(videofile):
            generate_frames(videofile) 
