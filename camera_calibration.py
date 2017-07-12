import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# pre-calculated values
'''
Camera matrix:
 [[  1.15396093e+03   0.00000000e+00   6.69705359e+02]
 [  0.00000000e+00   1.14802495e+03   3.85656232e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
Distortion coefficients:
 [[ -2.41017968e-01  -5.30720497e-02  -1.15810318e-03  -1.28318544e-04   2.67124303e-02]]
'''

# for showing found corners for each image
show_corner_images = False

img_points = [] # 2D points in image plane
obj_points = [] # 3d points in real world

# preparation of object points (same for all images)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)     # x, y coordinates, z = 0

# store calibration image names in a list
calib_images = glob.glob('./camera_cal/calibration*.jpg')

print("Loading images and finding chessboard corners...")
for img_name in calib_images:
    # load current image
    img = mpimg.imread(img_name)

    # convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6))

    # add object points and image points if corners found
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)

        if show_corner_images == True:
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.putText(img, img_name, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2 )
            cv2.imshow('Corners', img)
            cv2.waitKey(250)

# CAMERA CALIBRATION
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("done!\nCamera calibration successfull.")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
