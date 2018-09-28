import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle


from helper.helpers import *

# Get the camera cal data
mtx, dist = get_camera_cal()

# Get perpective transform parmeter
M, Minv = get_perspective_trans()

# Read image
image = mpimg.imread("test_images/test5.jpg")
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
wrap_image("output_images/threshed/", "output_images/wraped/", M, img_size)

# plt.imshow(image_warped, cmap="gray")
# plt.show()