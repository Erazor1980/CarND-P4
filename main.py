import camera_calibration as cb
import cv2
import numpy as np

#################
#### DEFINES ####
#################
use_precalculated_calib = True  # not wasting time each time with the same calibration procedure
test_camera_calib = False       # undistort and display a test image


############################
#### CAMERA CALIBRATION ####
############################
if use_precalculated_calib == True:
    mtx, dist = cb.get_precalculated_calibraton()
else:
    mtx, dist = cb.calibrate_camera(False)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

##########################
#### CALIBRATION TEST ####
##########################
if test_camera_calib == True:
    img = cv2.imread("./camera_cal/calibration1.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imshow('Distorted Image', img)
    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(0)

#######################
#### LOADING VIDEO ####
#######################
# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip

#TODO

############################
#### CORRECT DISTORTION ####
############################
img = cv2.imread('./test_images/test2.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)

# convert to grayscale
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

################################
#### THRESHOLD BINARY IMAGE ####
################################
import pipeline as pl
color_binary = pl.pipeline( undist )

cv2.imshow('test', undist)
cv2.imshow('th', color_binary)
cv2.waitKey(0)