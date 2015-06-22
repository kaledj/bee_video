''' 
Video Writer
============

Saves a processed video stream so it can be viewed without 
any recomputation. 
'''

import cv2
from video_loader import load_local
import kalman_track

def save_video(outputFilename):
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    # fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    # fourcc = cv2.VideoWriter_fourcc('M','P','4','2')
    writer = cv2.VideoWriter()
    writer.open(outputFilename, fourcc, 24, (640, 480))
    print(writer.isOpened())
    kt = kalman_track.App("../videos/newhive_noshadow3pm.h264", invisible=True)
    for frame in kt.run(False):
        writer.write(frame)


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


if __name__ == '__main__':
    save_video('testout.mkv')