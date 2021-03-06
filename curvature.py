import numpy as np
def calc_curvature( left_x, right_x, y_points ):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(y_points * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_points * ym_per_pix, right_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(y_points)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    meanCurve = ( left_curverad + right_curverad ) / 2.0
    imgCenter = 640
    distToCenter = ( ( right_x[-1] - left_x[-1] ) / 2.0 + left_x[-1] - imgCenter ) * xm_per_pix
    return meanCurve, distToCenter
