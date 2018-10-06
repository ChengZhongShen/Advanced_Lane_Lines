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
	pickle_file = open("./camera_cal/camera_cal.p", "rb")
	dist_pickle = pickle.load(pickle_file)
	mtx = dist_pickle["mtx"]  
	dist = dist_pickle["dist"]
	pickle_file.close()

	return mtx, dist

def get_perspective_trans():
	"""
	Read the pickle file and return the M, Minv to caller
	"""
	pickle_file = open("./helper/trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]  
	Minv = trans_pickle["Minv"]

	return M, Minv

def undistort_images(src, dst):
	"""
	undistort the images in src folder to dst folder
	"""
	# load dst, mtx
	pickle_file = open("../camera_cal/camera_cal.p", "rb")
	dist_pickle = pickle.load(pickle_file)
	mtx = dist_pickle["mtx"]  
	dist = dist_pickle["dist"]
	pickle_file.close()
	
	# loop the image folder
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


def wrap_images(src, dst):
	"""
	apply the wrap to images
	"""
	# load M, Minv
	img_size = (1280, 720)
	pickle_file = open("../helper/trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]
	Minv = trans_pickle["Minv"]
	# loop the file folder
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
		image_wraped = cv2.cvtColor(image_wraped, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_image, image_wraped)

if __name__ == '__main__':
	"""
	to run these function for challenge and harder, you should create the folders fist.
	"""
	# undistort_images("../test_images/", "../output_images/")
	# undistort_images("../test_images/challenge/", "../output_images/")
	# undistort_images("../test_images/harder/", "../output_images/")

	wrap_images("../output_images/", "../examples/")
	# wrap_images("../output_images/harder/threshed/", "../output_images/harder/wraped/")
	# wrap_images("../output_images/harder/undistort/", "../output_images/harder/wraped_color/")