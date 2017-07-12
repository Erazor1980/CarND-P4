import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# for showing found corners for each image
show_corner_images = True

img_points = [] # 2D points in image plane
obf_points = [] # 3d points in real world

# store calibration image names in a list
calib_images = glob.glob('./camera_cal/calibration*.jpg')

for img_name in calib_images:
    # load current image
    img = mpimg.imread(img_name)

    # convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6))

    # add object points and image points if corners found
    if ret == True:
        img_points.append(corners)
        #TODO obf_points

        if show_corner_images == True:
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.putText(img, img_name, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2 )
            cv2.imshow('Corners', img)
            cv2.waitKey(250)

