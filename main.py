import camera_calibration as cb
import cv2
import numpy as np

#################
#### DEFINES ####
#################
use_precalculated_calib = True  # not wasting time each run with the same calibration procedure
test_camera_calib = False       # undistort and display a test image
display_top_view = False        # show warped (top view) image (more for debugging)

test_on_single_image = True     # else the whole video will be processed

############################
#### CAMERA CALIBRATION ####
############################
if use_precalculated_calib == True:
    mtx, dist = cb.get_precalculated_calibraton()
else:
    mtx, dist = cb.calibrate_camera(False)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# CALIBRATION TEST
if test_camera_calib == True:
    img = cv2.imread("./camera_cal/calibration1.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imshow('Distorted Image', img)
    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(0)


def process_frame(img, display_result = False):
    ############################
    #### CORRECT DISTORTION ####
    ############################
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    ################################
    #### THRESHOLD BINARY IMAGE ####
    ################################
    import pipeline as pl
    color_binary, combined_binary = pl.pipeline( undist )

    ########################
    #### TOP VIEW IMAGE ####
    ########################
    import helper as hlp
    img_size = (undist.shape[1], undist.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    #print("src",src)

    dst = np.float32(
        [[(img_size[0] / 5), 0],
        [(img_size[0] / 5), img_size[1]],
        [(img_size[0] * 4 / 5), img_size[1]],
        [(img_size[0] * 4 / 5), 0]])
    #print("dst",dst)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    top_view_img = cv2.warpPerspective(combined_binary, M, img_size)

    #cv2.imwrite('test_binary.jpg', top_view_img)

    if display_top_view == True:
        warped = cv2.warpPerspective(color_binary, M, img_size)
        hlp.draw_polygon( warped, dst )
        hlp.draw_polygon( color_binary, src )
        cv2.imshow('test', color_binary)
        cv2.imshow('top_view', warped)
        cv2.waitKey(0)

    ########################
    #### FINDING LINES #####
    ########################
    import finding_lines as fl
    l_line = fl.Line()  # left line
    r_line = fl.Line()  # right line
    #fl.find_lines( np.uint8(top_view_img), l_line, r_line, True )
    fl.find_lines( top_view_img, l_line, r_line, True )

    ##########################
    #### DRAW FINAL IMAGE ####
    ##########################
    # Create an image to draw the lines on
    color_warp = np.zeros(undist.shape)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_line.current_fit_x, l_line.current_fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_line.current_fit_x, r_line.current_fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Given src and dst points, calculate the inverse perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, np.uint8(newwarp), 0.3, 0)

    if display_result == True:
        cv2.imshow('test', result)
        cv2.waitKey(0)
    return result


if test_on_single_image == True:
    img = cv2.imread('./test_images/test6.jpg')
    process_frame(img, True)
else:
    #######################################
    #### LOADING AND PROCESSING VIDEO  ####
    #######################################
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

    result_path = 'output/result_video.mp4'
    video = VideoFileClip("project_video.mp4")
    result_video = video.fl_image(process_frame) #NOTE: this function expects color images!!
    result_video.write_videofile(result_path, audio=False)