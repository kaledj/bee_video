'''
Feature generator
============

Computes the features of an image frame
'''

import cv2
import numpy as np

HAAR_KERNEL_SIZE = 12

def integralImg(image):
    imageGray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    return cv2.integral(imageGray)

def feature_all(image):
    pass

def feature_bgr(image):
    return image[:,:,0], image[:,:,1], image[:,:,2]

def feature_gradientOrientation(image):
    gradX, gradY = gradientXY(image)
    return np.arctan2(gradY, gradX)

def feature_gradientMagnitude(image):
    gradX, gradY = gradientXY(image)
    return np.sqrt(gradX.astype(np.int)**2 
                 + gradY.astype(np.int)**2).astype(np.uint8)

def gradientXY(image):
    imageGray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    cv2.GaussianBlur(imageGray, (3 ,3), 0, imageGray)

    kernelX = np.array([-1, 0, 1]).reshape((3, 1))
    kernelY = np.array([-1, 0, 1]).reshape((1, 3))

    gradX = cv2.filter2D(imageGray, -1, kernelX, 2) 
    gradY = cv2.filter2D(imageGray, -1, kernelY, 2) 

    return gradX, gradY

def feature_haar1(image):
    integral = integralImg(image)

    print integral.shape
    
    mid = (HAAR_KERNEL_SIZE / 2) -1
    bottom = HAAR_KERNEL_SIZE - 1
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[0][mid] = -2
    kernel[0][bottom] = 1
    kernel[bottom][0] = -1
    kernel[bottom][mid] = 2
    kernel[bottom][bottom] = -1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]

def feature_haar2(image):
    integral = integralImg(image)

    mid = (HAAR_KERNEL_SIZE / 2) - 1
    bottom = HAAR_KERNEL_SIZE - 1
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[0][bottom] = -1
    kernel[mid][0] = -2
    kernel[mid][bottom] = 2
    kernel[bottom][0] = 1
    kernel[bottom][bottom] = -1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]

def feature_haar3(image):
    integral = integralImg(image)

    midL = (HAAR_KERNEL_SIZE / 3) - 1
    midR = midL + (HAAR_KERNEL_SIZE / 3)
    bottom = HAAR_KERNEL_SIZE - 1    
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[0][midL] = -2
    kernel[0][midR] = 2
    kernel[0][bottom] = -1
    kernel[bottom][0] = -1
    kernel[bottom][midL] = 2
    kernel[bottom][midR] = -2 
    kernel[bottom][bottom] = 1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]    

def feature_haar4(image):
    integral = integralImg(image)

    midL = (HAAR_KERNEL_SIZE / 4) - 1
    midR = midL + (HAAR_KERNEL_SIZE / 2)
    bottom = HAAR_KERNEL_SIZE - 1     
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[0][midL] = -2
    kernel[0][midR] = 2
    kernel[0][bottom] = -1
    kernel[bottom][0] = -1
    kernel[bottom][midL] = 2
    kernel[bottom][midR] = -2 
    kernel[bottom][bottom] = 1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]    

def feature_haar5(image):
    integral = integralImg(image)

    midL = (HAAR_KERNEL_SIZE / 3) - 1
    midR = midL + (HAAR_KERNEL_SIZE / 3)
    bottom = HAAR_KERNEL_SIZE - 1     
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[midL][0] = -2
    kernel[midR][0] = 2
    kernel[bottom][0] = -1
    kernel[0][bottom] = -1
    kernel[midL][bottom] = 2
    kernel[midR][bottom] = -2
    kernel[bottom][bottom] = 1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]    

def feature_haar6(image):
    integral = integralImg(image)

    midL = (HAAR_KERNEL_SIZE / 4) - 1
    midR = midL + (HAAR_KERNEL_SIZE / 2)
    bottom = HAAR_KERNEL_SIZE - 1     
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[midL][0] = -2
    kernel[midR][0] = 2
    kernel[bottom][0] = -1
    kernel[0][bottom] = -1
    kernel[midL][bottom] = 2
    kernel[midR][bottom] = -2
    kernel[bottom][bottom] = 1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]    

def feature_haar7(image):
    integral = integralImg(image)

    midL = (HAAR_KERNEL_SIZE / 3) - 1
    midR = midL + (HAAR_KERNEL_SIZE / 3)
    bottom = HAAR_KERNEL_SIZE - 1     
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = 1
    kernel[0][bottom] = -1
    kernel[midL][midL] = -1
    kernel[midL][midR] = 1
    kernel[midR][midL] = 1
    kernel[midR][midR] = -1
    kernel[bottom][0] = -1
    kernel[bottom][bottom] = 1

    return cv2.filter2D(integral, -1, kernel)[1:, 1:]    

def feature_haar8(image):
    integral = integralImg(image)
    
    mid = (HAAR_KERNEL_SIZE / 2) - 1
    bottom = HAAR_KERNEL_SIZE - 1
    kernel = np.zeros((HAAR_KERNEL_SIZE, HAAR_KERNEL_SIZE))
    kernel[0][0] = -1
    kernel[0][mid] = 2
    kernel[0][bottom] = -1
    kernel[mid][0] = 2
    kernel[mid][mid] = -4
    kernel[mid][bottom] = 2
    kernel[bottom][0] = -1
    kernel[bottom][mid] = 2
    kernel[bottom][bottom] = -1
    
    return cv2.filter2D(integral, -1, kernel)[1:, 1:]



if __name__ == '__main__':
    pass
