import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle


from helper.helpers import *
from helper.lane_detection import *
from helper.cal_curv import measure_curv, measure_offset

# Get the camera cal data
mtx, dist = get_camera_cal()

# Get perpective transform parmeter
M, Minv = get_perspective_trans()

# Read image
image = mpimg.imread("test_images/test6.jpg")
img_size = (image.shape[1], image.shape[0])

# distort the image
image_dist = cv2.undistort(image, mtx, dist, None, mtx)

# Undistort the images in test image folder
# undistort_images("test_images/", "output_images/undistort/", mtx, dist)

# apply thresholding
s_thresh=(170,255)
sx_thresh=(20, 100)
image_threshed = color_grid_thresh(image_dist, s_thresh=s_thresh, sx_thresh=sx_thresh)

# thresh the images in the undistort image folder
# thresh_images("output_images/undistort/", "output_images/threshed/", s_thresh, sx_thresh)

# apply the view_perspective tranform
image_warped = cv2.warpPerspective(image_threshed, M, img_size, flags=cv2.INTER_LINEAR)

# apply the warp to threshed images
# wrap_images("output_images/threshed/", "output_images/wraped/", M, img_size)

image_fit_line = fit_polynomial(image_warped*255)
# apply the line fit to 

leftx, lefty, rightx, righty, out_img = find_lane_pixels(image_warped)

left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
offset = measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
print(left_curverad, right_curverad)
print(offset)

# plt.imshow(image_fit_line)
# plt.show()