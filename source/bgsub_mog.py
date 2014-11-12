'''
Background subtraction using MOG

'''

import numpy as np
import cv2, os, sys
import segmentation, features, keys

def bgsub(vidfilename):
    video = cv2.VideoCapture(vidfilename)
    prevret, prevframe = video.read()
    ret, frame = video.read()
    operator1 = cv2.BackgroundSubtractorMOG2(100,256,True)
    operator2 = cv2.BackgroundSubtractorMOG2(100,256,True)

    # print dir(operator1)
    # print operator1.getAlgorithm
    # exit()
    mask = np.zeros_like(prevframe)
    color = np.random.randint(0,255,(100,3))
    while ret:

        mask1 = operator1.apply(frame, learningRate = .50)
        mask2 = operator2.apply(frame, learningRate = -1)
        #mask2 = 1 - mask2

        diff = mask1.astype(np.int) - mask2.astype(np.int)
        diff = np.abs(diff).astype(np.uint8)
        
        cv2.imshow("Mask1", mask1)

        contours, hierarchy = cv2.findContours((mask2.copy()),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
        contours = np.vstack(contours).squeeze()
        contours = contours.astype(np.float32)
        cv2.imshow("Mask1", mask1)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask2", mask2)
        cv2.imshow("Differences", diff)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prevframe,frame, contours)
        # Select good points
        good_new = p1[st==1]
        good_old = contours[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new[0:99],good_old[0:99])):
            print new.size
            print old.size
            if new.size <=   1: continue
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow("Test", img)
        ret, frame = video.read()
        key = cv2.waitKey(5)
        if key == 32: break
        elif key == 27: 
            exit()


if __name__ == '__main__':
    bgsub('../videos/whitebg.h264')
    sys.exit()

    videos = []
    for filename in os.listdir("../videos"):
        videos.append("../videos/" + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            print videofile
            bgsub(videofile)    
            