'''
Background subtraction using MOG

'''

import time
import numpy as np
import os
import sys
import cv2
import keys

# User modules
from analysis import *

BGR2GRAY = cv2.cv.CV_BGR2GRAY
GT_IMG_DIR = 'C:/Users/kaledj/Projects/SegmentationforCortina/images/'
VIDEO_DIR = '../videos/'


def bgsub(vidfile_basename, threshold):
    time_i = time.time()
    operator = cv2.BackgroundSubtractorMOG2(2000, threshold, True)
    # Learn the bg
    model_bg2(VIDEO_DIR + vidfile_basename, operator)

    tp_t = fp_t = p_t = n_t = 0

    video = cv2.VideoCapture(VIDEO_DIR + vidfile_basename)
    ret, frame = video.read()
    frame_num = 0
    while ret:
        mask = operator.apply(frame, learningRate=-1)
        mask = morph_openclose(mask)
        mask_binary = (mask == 255).astype(np.uint8)

        gt_filename = "{0}/{1}/{2}.jpg.seg.bmp".format(GT_IMG_DIR, vidfile_basename, frame_num)
        if os.path.exists(gt_filename):
            cv2.imshow("Ground truth", cv2.imread(gt_filename) * 255)
            tp, fp = compare_response_to_truth(mask_binary, gt_filename)
            # print("True Pos: {0}\nFalse Pos: {1}".format(tp, fp))
            pos_detected, neg_detected = class_counter.count_posneg(mask_binary)
            tp_t += tp
            fp_t += fp
            p_t += pos_detected
            n_t += neg_detected
            # print("Foreground pixels: {0}\nBackground pixels: {1}".format(pos_detected, neg_detected))

        mask = ((mask == 255) * 255).astype(np.uint8)
        #cv2.imshow("Mask", mask)
        blob_detect(mask, frame)

        ret, frame = video.read()
        frame_num += 1
        key = cv2.waitKey(1)
        if key == keys.ESC:
            break
        if key == keys.SPACE:
            cv2.waitKey()
        if key == keys.Q:
            exit()

    print("{0} seconds elapsed.".format(time.time() - time_i))
    if p_t and n_t:
        tp_r = float(tp_t) / p_t
        fp_r = fp_t / float(n_t)
        print tp_t, p_t, fp_t, n_t
        print("TPR: {0}\tFPR: {1}".format(tp_r, fp_r))
        return tp_r, fp_r


def model_bg(video):
    vidcapture = cv2.VideoCapture(video)
    ret, frame = vidcapture.read()
    frame = cv2.cvtColor(frame, BGR2GRAY)
    vres, hres = frame.shape

    # Average first N frames
    N = 100
    frameBuffer = np.zeros((N, vres, hres), np.uint32)
    for i in range(N):
        ret, frame = vidcapture.read()
        if ret:
            frame = cv2.cvtColor(frame, BGR2GRAY)
            frameBuffer[i] = frame
        else:
            break
    vidcapture.release()
    return (np.sum(frameBuffer, axis=0) / N).astype(np.uint8)


def model_bg2(video, operator):
    vidcapture = cv2.VideoCapture(video)
    # Initialize from first N frames
    N = 50
    for _ in range(N):
        ret, frame = vidcapture.read()
        if ret:
            operator.apply(frame, learningRate=-1)
        else:
            break
    vidcapture.release()


def morph_openclose(image):
    kernel = np.ones((3, 3), np.uint8)
    new_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel, iterations=1)


def blob_detect(mask, frame):
    contours, hierarchy = cv2.findContours((mask.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Frame", frame)




if __name__ == '__main__':
    test_videoname = "whitebg.h264"
    # bgsub('../videos/whitebg.h264')
    # sys.exit()

    detection_rates = {'tp': [], 'fp': []}
    # for i in xrange(10):
    #     print("Testing with threshold {0}".format(2**i))
    #     tp_rate, fp_rate = bgsub(os.path.basename(test_videoname), 2**i)
    #     detection_rates['tp'].append(tp_rate)
    #     detection_rates['fp'].append(fp_rate)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(detection_rates['fp'], detection_rates['tp'])
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
    # print(detection_rates)
    # exit()

    videos = []
    for filename in os.listdir(VIDEO_DIR):
        videos.append(VIDEO_DIR + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            print("Opening: {0} ({1})".format(os.path.basename(videofile), videofile))
            try:
                tp_rate, fp_rate = bgsub(os.path.basename(videofile), 16)
                detection_rates['tp'].append(tp_rate)
                detection_rates['fp'].append(fp_rate)
            except TypeError:
                print("No ground truths for {0} yet.".format(videofile))
