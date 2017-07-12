import camera_calibration as cb
import numpy as np

#mtx, dist = cb.calibrate_camera(False)

mtx, dist = cb.get_precalculated_calibraton()
print(mtx)
print(dist)