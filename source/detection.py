'''
Detection
=========

Implements multiple different types of detection. HAAR, SIFT...
'''

# Local modules
from video_loader import load_local
import keys
# System modules
import cv2, os, sys
import numpy as np

ROI = (100, 250)
ROI_W = 370
ROI_H = 200

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)

def cascadeDetect(vidfilename, min_neighbors=2):
    cascade = cv2.CascadeClassifier("../classifier/v2verticaldown/cascade.xml")
    print videofile
    video = load_local(vidfilename)
    ret, frame = video.read()
    while ret:
        # frameGray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        bees = cascade.detectMultiScale(frame, minNeighbors=min_neighbors)
        for (x, y, w, h) in bees:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            center = (x+(w/2), y+(h/2))
        cv2.rectangle(frame, ROI, (ROI[0]+ROI_W, ROI[1]+ROI_H), (0, 255, 0), 2)
        cv2.imshow("Video", frame)
        ret, frame = video.read()
        key = cv2.waitKey(1)
        if key == keys.ESC:
            break
        if key == keys.SPACE:
            cv2.waitKey()
        if key == keys.Q:
            exit()


def siftDetect(vidfilename):
    video = load_local(vidfilename)
    ret, frame = video.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        
        sift = cv2.SIFT()
        kp = sift.detect(frame, None)
        print(type(kp[0]))
        frame = cv2.drawKeypoints(frame, kp)
        cv2.imshow("Keypoints", frame)

        ret, frame = video.read()
        key = cv2.waitKey(10)
        if key == 32: break
        elif key == 27: 
            exit()

def surfDetect(vidfilename):
    video = load_local(vidfilename)
    ret, frame = video.read()
    prevFrame = frame.copy()
    prevFrame = cv2.cvtColor(prevFrame, cv2.cv.CV_BGR2GRAY)

    surf = cv2.SURF(100, upright=False)
    bgSub = cv2.BackgroundSubtractorMOG()
    kpMatcher = cv2.BFMatcher()
    while ret:
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

        fgMask0 = bgSub.apply(prevFrame)
        fgMask1 = bgSub.apply(frame)
        # if not np.any(fgMask0): fgMask0 = None
        # if not np.any(fgMask1): fgMask1 = None
        kp0 = surf.detect(prevFrame, fgMask0)
        kp1 = surf.detect(frame, fgMask1)
        kp0, des0 = surf.compute(prevFrame, kp0)
        kp1, des1 = surf.compute(frame, kp1)
        matches = kpMatcher.match(des0, des1)
        matches = sorted(matches, key = lambda x:x.distance)

        drawMatches(prevFrame, kp0, frame, kp1, matches)
        # cv2.imshow("Keypoints", frames)

        prevFrame = frame.copy()
        ret, frame = video.read()
        key = cv2.waitKey(1)
        if key is keys.SPACE: cv2.waitKey()
        elif key is keys.ESC: 
            exit()

def cross(pt0, pt1, rect):
    ''' 
    Determines if the points enter or leave the rectangle.
    Returns 1 if the points leave the rect, -1 if it enters, or 0 if neither.
    '''
    pt0Bool = pt0[0]>=rect.corner[0] and pt0[0] <= rect.corner[0]+rect.w and pt0[1]>=rect.corner[1] and pt0[1]<=rect.corner[1]+rect.h
    pt1Bool = pt1[0]>=rect.corner[0] and pt1[0] <= rect.corner[0]+rect.w and pt1[1]>=rect.corner[1] and pt1[1]<=rect.corner[1]+rect.h
    if not pt0Bool and pt1Bool:
        return 1
    elif pt0Bool and not pt1Bool:
        return -1
    else:
        return 0

def exit():
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    videos = []
    videos.append("../videos/newhive_noshadow3pm.h264")
    # for filename in os.listdir("../videos"):
    #     videos.append("../videos/" + filename)
    for videofile in videos:
        if os.path.isfile(videofile):
            cascadeDetect(videofile, min_neighbors=4)    
        else:
            print("%s not found." % videofile)
