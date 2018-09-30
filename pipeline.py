import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

from helper.helpers import get_camera_cal, get_perspective_trans
from helper.image_process import color_grid_thresh, draw_lane
from helper.lane_detection import find_lane_pixels, get_polynomial
from helper.cal_curv import measure_curv, measure_offset


mtx, dist = get_camera_cal()
M, Minv = get_perspective_trans()

def pipeline(image):
	'''
	pipline without tracker
	'''
	img_size = (image.shape[1], image.shape[0])

	# distort the image
	image_undist = cv2.undistort(image, mtx, dist, None, mtx)

	# apply thresholding
	s_thresh=(170,255)
	sx_thresh=(20, 100)
	image_threshed = color_grid_thresh(image_undist, s_thresh=s_thresh, sx_thresh=sx_thresh)

	# apply the view_perspective tranform
	image_warped = cv2.warpPerspective(image_threshed, M, img_size, flags=cv2.INTER_LINEAR)

	# find lane pixels
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(image_warped)

	# measure the curverad and offset
	left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
	curverad = (left_curverad + right_curverad) / 2
	offset = measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)

	# get the polynomial line points
	left_fitx, right_fitx, ploty = get_polynomial(leftx, lefty, rightx, righty, img_size)

	# draw the lane to the undist image
	result = draw_lane(image_undist, image_warped, Minv, left_fitx, right_fitx, ploty)

	return result

##################################################
def one_image_test():

	image = mpimg.imread("test_images/test6.jpg")

	result = pipeline(image)
	plt.imshow(result)
	plt.show()

def images_test(src, dst):
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print("handle on: ", file)
		img = mpimg.imread(file)
		result = pipeline(img)
		file_name = file.split("\\")[-1]
		# print(file_name)
		out_image = dst+file_name
		# print(out_image)
		image_dist = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_image, image_dist)

# one_image_test()
images_test("test_images/", "output_images/")

## in process test
# thresh the images in the undistort image folder
# thresh_images("output_images/undistort/", "output_images/threshed/", s_thresh, sx_thresh)

# Undistort the images in test image folder
# undistort_images("test_images/", "output_images/undistort/", mtx, dist)

# apply the warp to threshed images
# wrap_images("output_images/threshed/", "output_images/wraped/", M, img_size)

# image_fit_line = fit_polynomial(image_warped*255)
# apply the line fit to 