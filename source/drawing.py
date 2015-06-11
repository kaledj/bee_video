import cv2
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

def draw_frame_num(frame, num):
    params = dict(fontFace=cv2.cv.CV_FONT_HERSHEY_COMPLEX,
                  fontScale=1, thickness=1)
    ret, baseline = cv2.getTextSize(str(num), **params)
    cv2.putText(frame, str(num), org=(0, ret[1]), color=RED, **params)

def draw_prev_points(frame, points, color=BLUE, radius=2):
    if points is not None and len(points) > 0:
        p0 = np.float32([point for point in points]).reshape(-1, 1, 2)
        if p0 is not None:
            for (x, y) in p0.reshape(-1, 2):
                cv2.circle(frame, (x, y), radius=radius, color=color, thickness=-1)
        return True
    else:
        return False

def draw_contours(frame, fg_mask):
    assert frame is not None
    if cv2.__version__ == '3.0.0':
        _, contours, hierarchy = cv2.findContours((fg_mask.copy()), 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    else:
        contours, hierarchy = cv2.findContours((fg_mask.copy()), 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), hierarchy=hierarchy, maxLevel=2)
    return contours, hierarchy


def draw_min_ellipse(contours, frame, minArea, maxArea):
    areas = []
    centers = []
    for contour in contours:
        if len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        # print(cv2.minAreaRect(contour))
        width, height = ellipse[1]
        area = (width / 2) * (height / 2) * np.pi
        # print("Area: " + str(area))
        # print("Size: ", ellipse[1])
        # print("Center: ", ellipse[0])
        areas.append(area)
        if minArea < area < maxArea:
            cv2.ellipse(frame, ellipse, color=BLUE)
            center = tuple([int(x) for x in ellipse[0]])
            cv2.circle(frame, center, radius=4, color=BLUE, thickness=-1)
            centers.append(center)
    return areas, centers


def draw_blob_centers(contours, hierarchy, frame=None, drawcenters=False):
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        center = (x + (w/2), y + (h/2))
        if drawcenters and frame is not None:
            cv2.circle(frame, center, radius=4, color=RED, thickness=-1)
            if len(contour) >= 5:
                center2 = cv2.fitEllipse(contour)[0]
                center2 = tuple([int(x) for x in center2])
                cv2.circle(frame, center2, radius=4, color=BLUE, thickness=-1)
        centers.append(center)
    return centers


def draw_rectangle(frame, topLeft, bottomRight, color=GREEN, thickness=2):
    cv2.rectangle(frame, topLeft, bottomRight, color, thickness)


def draw_line(frame, fromPointTuple, toPointTuple, color=RED):
    cv2.line(frame, fromPointTuple, toPointTuple, color)
