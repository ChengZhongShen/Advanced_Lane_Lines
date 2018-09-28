import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

from helpers import *

# Read the pickle file
pickle_file = open("camera_cal/camera_cal.p", "rb")
dist_pickle = pickle.load(pickle_file)
mtx = dist_pickle["mtx"]  
dist = dist_pickle["dist"]

# Read image
image = cv2.imread("test_images/test5.jpg")

# distort the image
image_dist = cv2.undistort(image, mtx, dist, None, mtx)

# Undistort the images in test image folder
image_files = glob.glob("test_images/*.jpg")
for idx, file in enumerate(image_files):
	print(file)
	img = cv2.imread(file)
	image_dist = cv2.undistort(img, mtx, dist, None, mtx)
	file_name = file.split("\\")[-1]
	print(file_name)
	out_image = "output_images/undistort/"+file_name
	print(out_image)
	cv2.imwrite(out_image, image_dist)


# # apply thresholding
# s_thresh=(170,255)
# sx_thresh=(20, 100)
# image_threshed = color_grid_thresh(image_dist, s_thresh=s_thresh, sx_thresh=sx_thresh)

# plt.imshow(image_threshed, cmap='gray')
# plt.show()