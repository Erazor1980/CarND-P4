## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/camCalibExample.png "Undistorted Chessboard"
[image2]: ./examples/undistImg.png "Undistorted Image"
[image3]: ./examples/binaryImg.png "Binary Image"
[image4]: ./examples/topView.png "Top View Image"
[image5]: ./examples/findingLines.png "Finding Lines"
[image6]: ./examples/backPlot.png "Green Area Between Lines"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

The code for this step is contained in the camera_calibration.py. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the pipeline method in `pipeline.py`).  Here's an example of my output for this step.

![alt text][image3]

The code for my perspective can be found below the "TOP VIEW IMAGE" comment (lines 50ff). I chose the hardcode the source and destination points in the following manner:

```python
img_size = (undist.shape[1], undist.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 5), 0],
    [(img_size[0] / 5), img_size[1]],
    [(img_size[0] * 4 / 5), img_size[1]],
    [(img_size[0] * 4 / 5), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 256, 0        | 
| 203, 720      | 256, 720      |
| 1127, 720     | 1024, 720      |
| 695, 460      | 1024, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

You can test it in code setting the "display_top_view" to "True". I have implemented a draw_polygon function (in helper.py), which I used to draw the points/lines.

Then I implemented find_lines function (see finding_lines.py), which performs following steps to find lanes in the image:
* calculate a column wise histogram for the bottom half of the binary threshold image to find 2 peaks (= starting points of the lines)
* use sliding window approach (15 rectangles per line) to find next line parts upwards 
* collect points within these windows (for left and right line)
* fit lane lines with a 2nd order polynomial kinda like this

![alt text][image5]

As in the most of my python scripts, which are used in the main.py, there is a __main__ in which can be used to see some test output.

In the curvature.py I implemented the function "calc_curvature", which additionally calculates the distance of the car to the center between the lines. Both values are returned by this function. 

First I approximately calculate the meters per pixel in both directions. Then I fin new polynomials through the points in world space. Next two curves (for left and right line) are calculated as described in the udacity lesson. The mean value of both is the output value.
For the distance to the center I calculate the difference between the center between the two lines (bottom part) and the image center. And then multiply this value by xm_per_pix.

Both values (curvature and distance to center in meter) are displayed to the result image. Below the comment "DRAW FINAL IMAGE" you can find the implementation of the final drawing, the green polygon between the 2 detected lines. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./output/result_video.mp4)

---

### Discussion

Right now there is now "temporal" tracking of the lines implemented, or any kind of filtering. The lines are detected frame-wise. This could cause problems, when e.g. some other peaks appear in the image (dirty road, other car, etc.). 

A very nice improvement would be to track the history of the line, and smooth the results. Search only near the last stable lines found etc.

