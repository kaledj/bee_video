'''
Generate Frames
===============

Takes every 10th frame from a video sequence to be used in cropping out samples. 
'''

import cv2

def generate_frames(filename, prefix):
    vidcapture = cv2.VideoCapture(filename)
    for i in range(500):
        ret, frame = vidcapture.read()
        if ret and (i % 50 == 0):
            print cv2.imwrite("samples/positive/test/"
                + prefix + str(i / 50) + ".png", frame)
        if not ret: 
            break

def main():
    generate_frames("videos/whitebg.h264", 'a')
    # generate_frames("videos/9-2_rpi9.mkv", 'b')
    # generate_frames("videos/724video.mkv", 'c')
    # generate_frames("videos/video1.mkv", 'd')

if __name__ == '__main__':
    main()
