'''
Split Frames
===============

Takes every 10th frame from a video sequence to be used in cropping out samples. 
'''

import sys
# PREPEND the source to the path so that this package can access it
sys.path.insert(0, 'C:/Users/kaledj/Projects/bee_video_dev')

from source import keys
import cv2
import os
import time
from multiprocessing import Process, Lock
from collections import deque

VIDEO_DIR = '../videos/'
OUTPUT_DIR_BASE = 'C:/Users/kaledj/Projects/SegmentationforCortina/images/'
FRAME_STRIDE = 50


def print_locked(lock, string):
    lock.acquire()
    print(string)
    lock.release()


def generate_frames(lock, filename):
    vidcapture = cv2.VideoCapture(filename)

    base_filename = os.path.basename(filename)
    output_dir = OUTPUT_DIR_BASE + base_filename
    if not os.path.exists(output_dir):
        print_locked(lock, "Creating directory {0}".format(output_dir))
        os.mkdir(output_dir)

    ret = True
    idx = 0
    writes_t = writes_skipped = writes_made = 0
    while ret:
        ret, frame = vidcapture.read()
        if ret and idx % FRAME_STRIDE is 0:
            writes_t += 1
            out_filename = "{0}/{1}.jpg".format(output_dir, idx)
            if os.path.exists(out_filename):
                writes_skipped += 1
            else:
                writes_made += 1
                cv2.imwrite(out_filename, frame)
        idx += 1
        key = cv2.waitKey(1)
        if key is keys.ESC:
            print_locked(lock, "Stopping {0} early at frame {1}".format(base_filename, idx))
            break
        if key is keys.Q:
            print_locked(lock, "Quitting. Currently in {0} at frame {1}".format(base_filename, idx))
            exit()
    print_locked(lock, "{0} frames read. {1} skipped, {2} newly written in directory {3}.".format(
        writes_t, writes_skipped, writes_made, base_filename))


def main():
    cv2.namedWindow("Interface")

    # Get list of videos to process
    videos = []
    for filename in os.listdir(VIDEO_DIR):
        videos.append(VIDEO_DIR + filename)

    procs = deque()
    lock = Lock()

    # Process
    for videofile in videos:
        if os.path.isfile(videofile):
            print("Splitting {0} into frames.".format(videofile))
            proc = Process(target=generate_frames, args=(lock, videofile, ), name=videofile)
            procs.append(proc)
            proc.start()
    for proc_obj, join_func in ((proc, proc.join) for proc in procs):
        join_func()
        print("Process '{0}' joined.".format(proc_obj.name))


if __name__ == '__main__':
    time.clock()
    main()
    print("Time elapsed: {0} seconds".format(time.clock()))
