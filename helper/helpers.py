import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

def get_camera_cal():
	"""
	Read the picke file and return the mtx, dist to caller
	"""
	pickle_file = open("camera_cal/camera_cal.p", "rb")
	dist_pickle = pickle.load(pickle_file)
	mtx = dist_pickle["mtx"]  
	dist = dist_pickle["dist"]

	return mtx, dist

def get_perspective_trans():
	"""
	Read the pickle file and return the M, Minv to caller
	"""
	pickle_file = open("helper/trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]  
	Minv = trans_pickle["Minv"]

	return M, Minv

def undistort_images(src, dst, mtx, dist):
	"""
	undistort the images in src folder to dst folder
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_dist = cv2.undistort(img, mtx, dist, None, mtx)
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		image_dist = cv2.cvtColor(image_dist, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_image, image_dist)

def thresh_images(src, dst, s_thresh, sx_thresh):
	"""
	apply the thresh to images in a src folder and output to dst foler
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_threshed = color_grid_thresh(img, s_thresh=s_thresh, sx_thresh=sx_thresh)
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		# convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
		image_threshed = cv2.cvtColor(image_threshed*255, cv2.COLOR_GRAY2RGB)
		cv2.imwrite(out_image, image_threshed)

def wrap_images(src, dst, M, img_size):
	"""
	apply the wrap to images
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_wraped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		# no need to covert RGB to BGR since 3 channel is same
		cv2.imwrite(out_image, image_wraped)
