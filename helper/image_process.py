# File include the image process functions

# Function spec
# 1. input: the imput image is a RGB image, all other threshold parameter should have a default value
# 2. output: a binary(!!!) image which has the same size with input image

import numpy as np 
import cv2


def color_grid_thresh(img, s_thresh=(170,255), sx_thresh=(20, 100)):
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivateive in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# combine the two binary
	binary = sxbinary | s_binary

	# Stack each channel (for visual check the pixal sourse)
	# color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary,s_binary)) * 255
	return binary

def color_grid_thresh_dynamic(img, s_thresh=(170,255), sx_thresh=(20, 100)):
	img = np.copy(img)
	height = img.shape[0]
	width = img.shape[1]
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivateive in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	sxbinary[:, :width//2] = 0	# use the left side
	s_binary[:,width//2:] = 0 # use the right side

	# combine the two binary
	binary = sxbinary | s_binary

	# Stack each channel (for visual check the pixal sourse)
	# color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary,s_binary)) * 255
	return binary

def yellow_grid_thresh(img, y_low=(10,50,0), y_high=(30,255,255), sx_thresh=(20, 100)):
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivateive in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# # Threshold color channel
	# s_binary = np.zeros_like(s_channel)
	# s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	
	yellow_filtered = yellow_filter(img, y_low, y_high)
	yellow_filtered[yellow_filtered > 0] = 1 # transfer to binary

	# combine the two binary, right and left
	sxbinary[:,:640] = 0 # use right side of sxbinary
	yellow_filtered[:,640:] = 0 # use left side of yellow filtered


	binary = sxbinary | yellow_filtered

	# Stack each channel (for visual check the pixal sourse)
	# color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary,s_binary)) * 255
	return binary

def yellow_white_thresh(img, y_low=(10,50,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255)):
	
	yellow_filtered = yellow_filter(img, y_low, y_high)
	yellow_filtered[yellow_filtered > 0] = 1 # transfer to binary

	white_filtered = white_filter(img, w_low, w_high) # transfer to binary
	white_filtered[white_filtered > 0] = 1

	# combine the two binary, right and left
	yellow_filtered[:,640:] = 0 # use left side of yellow filtered
	white_filtered[:, :640] = 0 # use the right side of white filtered

	# plt.figure(),plt.imshow(yellow_filtered, cmap="gray")
	# plt.figure(),plt.imshow(white_filtered, cmap="gray")
	# plt.show()

	binary = yellow_filtered | white_filtered

	return binary

def y_w_dynamic(img, y_low=(10,0,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255)):
	"""
	auto adjust the y_low(_,_,*), V channel value according the image brightness
	auto adjust the w_low(*,*,*) according the image brightness
	"""
	height = img.shape[0]
	width = img.shape[1]
	y_offset = 0	# the yellow offset add to mean brightness
	w_offset = 30	# the white offset add to mean brightness
	HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	V = HSV[:,:,2]
	bright_lb = int(np.mean(V[height//2:, :width//2]))	# use left bottom corner brightness as ref
	bright_rb = int(np.mean(V[height//2:, width//2:]))	# use right bottom corner brightness as ref

	y_low = (10, 0, bright_lb+y_offset)	# set the yellow V low according the brightness
	
	w_low_thresh = min(bright_rb+w_offset, 255-10) # solve the problem that the valve > 255
	w_low = (w_low_thresh, w_low_thresh, w_low_thresh)	# set the white low valuse according the brightness

	sx_thresh=(30, 120)
	# if bright_rb > 120:
	# 	sx_thresh = (bright_rb - 80, bright_rb)
	sobelx = cv2.Sobel(V, cv2.CV_64F, 1, 0) # Take the derivateive in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	yellow_filtered = yellow_filter(img, y_low, y_high)
	yellow_filtered[yellow_filtered > 0] = 1 # transfer to binary

	# white_filtered = white_filter(img, w_low, w_high) # transfer to binary
	# white_filtered[white_filtered > 0] = 1

	# just for debug to check the threshold value
	# plt.figure(),plt.imshow(img)
	# plt.figure(),plt.imshow(yellow_filtered, cmap="gray")
	# plt.figure(),plt.imshow(white_filtered, cmap="gray")
	# plt.show()

	# combine the two binary, right and left
	yellow_filtered[:,640:] = 0 # use left side of yellow filtered
	# white_filtered[:, :640] = 0 # use the right side of white filtered
	sxbinary[:,:640] = 0
	

	# binary = yellow_filtered | white_filtered
	binary = yellow_filtered | sxbinary

	# # apply edage founding to avoid block white area.
	# binary = cv2.Laplacian(binary,cv2.CV_64F) # use floating piont to get postive and negetive gradient
	# binary = np.absolute(binary) # transfer to abs value
	# binary = np.uint8(binary)	# transfer to uint8
	# binary[binary > 0] = 1 # transfer to binary again


	return binary

def draw_lane_fit(undist, warped ,Minv, left_fitx, right_fitx, ploty):
	# Drawing
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

	# Warp the blank back to original image space using inverse perspective matrix(Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result

def draw_lane_find(undist, warped, Minv, leftx, lefty, rightx, righty):
	
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	color_warp[lefty, leftx] = [255, 0, 0]
	color_warp[righty, rightx] = [0 ,0 , 255]

	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result

def yellow_filter(image, low=(10,50,0), high=(30,255,255)):
	"""
	filter the right side yellow line out
	"""
	image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	image_filtered = cv2.inRange(image_HSV, low, high)

	return image_filtered

def white_filter(image, low=(0,0,0), high=(255,255,255)):
	"""
	"""
	image_filtered = cv2.inRange(image, low, high)

	return image_filtered

#####################################################################################
def test_draw():
	from lane_detection import find_lane_pixels, get_polynomial
	
	# read undist image
	undist = mpimg.imread("../output_images/undistort/test6.jpg")
	# read warped image
	binary_warped = mpimg.imread('../output_images/wraped/test6.jpg')
	binary_warped = binary_warped[:,:,0] # get 1 channel, three channel is same

	# get parameters
	img_size = (undist.shape[1], undist.shape[0])
	
	# has problem use the function in helper
	pickle_file = open("./trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]  
	Minv = trans_pickle["Minv"]

	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
	left_fitx, right_fitx, ploty = get_polynomial(leftx, lefty, rightx, righty, img_size)


	result_draw_fit = draw_lane_fit(undist, binary_warped ,Minv, left_fitx, right_fitx, ploty)
	result_draw_found = draw_lane_find(undist, binary_warped, Minv, leftx, lefty, rightx, righty)

	result_draw = draw_lane_find(result_draw_fit, binary_warped, Minv, leftx, lefty, rightx, righty)


	plt.figure(),plt.imshow(undist), plt.title("Undistort Image")
	plt.figure(),plt.imshow(result_draw_fit), plt.title("Draw fit lane")
	plt.figure(),plt.imshow(result_draw_found),plt.title("Draw found pixel")
	plt.figure(),plt.imshow(result_draw),plt.title("Draw found & fit")

	plt.show()


def test_thresh_images(src, dst, s_thresh, sx_thresh):
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

def test_color_grid_thresh_dynamic(src, dst, s_thresh, sx_thresh):
	"""
	apply the thresh to images in a src folder and output to dst foler
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_threshed = color_grid_thresh_dynamic(img, s_thresh=s_thresh, sx_thresh=sx_thresh)
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		# convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
		image_threshed = cv2.cvtColor(image_threshed*255, cv2.COLOR_GRAY2RGB)
		cv2.imwrite(out_image, image_threshed)

def test_yellow_grid_thresh_images(src, dst, y_low=(10,50,0), y_high=(30,255,255), sx_thresh=(20, 100)):
	"""
	apply the thresh to images in a src folder and output to dst foler
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_threshed = yellow_grid_thresh(img, y_low, y_high, sx_thresh)
		
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		# convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
		image_threshed = cv2.cvtColor(image_threshed*255, cv2.COLOR_GRAY2RGB)
		cv2.imwrite(out_image, image_threshed)

def test_yellow_white_thresh_images(src, dst, y_low=(10,50,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255)):
	"""
	apply the thresh to images in a src folder and output to dst foler
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		image_threshed = yellow_white_thresh(img, y_low, y_high, w_low, w_high)
		
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		# convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
		image_threshed = cv2.cvtColor(image_threshed*255, cv2.COLOR_GRAY2RGB)
		
		# HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		# V = HSV[:,:,2]
		# brightness = np.mean(V)
		# info_str = "brightness is: {}".format(int(brightness))
		# cv2.putText(image_threshed, info_str, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
		
		cv2.imwrite(out_image, image_threshed)

def test_y_w_dynamic_images(src, dst, y_low=(10,50,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255)):
	"""
	apply the thresh to images in a src folder and output to dst foler
	"""
	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		
		image_threshed = y_w_dynamic(img, y_low, y_high, w_low, w_high)
		
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = dst+file_name
		print(out_image)
		
		# convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
		image_threshed = cv2.cvtColor(image_threshed*255, cv2.COLOR_GRAY2RGB)
		
		# Caculate the brightness and write to image
		height = img.shape[0]
		width = img.shape[1]
		y_offset = 50	# the offset add to mean brightness
		w_offset = 50	# the offset add to mean brightness
		HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		V = HSV[:,:,2]
		bright_lb = int(np.mean(V[height//2:, :width//2]))	# use left bottom corner brightness as ref
		bright_rb = int(np.mean(V[height//2:, width//2:]))	# use right bottom corner brightness as ref
		info_str1 = "brightness is: {}".format(bright_lb)
		info_str2 = "brightness is: {}".format(bright_rb)
		# cv2.putText(image_threshed, info_str1, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
		# cv2.putText(image_threshed, info_str2, (690,700), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
		
		cv2.imwrite(out_image, image_threshed)
		# break

def test_thresh_image(image, s_thresh, sx_thresh):
	"""
	adjust the thresh parameters
	"""
	img = mpimg.imread(image)
	img_threshed = color_grid_thresh(img, s_thresh=s_thresh, sx_thresh=sx_thresh)

	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivateive in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# combine the two binary
	binary = sxbinary | s_binary

	
	plt.figure(),plt.imshow(img),plt.title("original")
	plt.figure(),plt.imshow(sxbinary, cmap='gray'),plt.title("x-gradient")
	plt.figure(),plt.imshow(s_binary, cmap='gray'),plt.title("color-threshed")
	plt.figure(),plt.imshow(s_channel, cmap='gray'),plt.title("s_channel")

	plt.figure(),plt.imshow(img_threshed, cmap='gray'),plt.title("combined-threshed")
	plt.show()

def test_yellow_filter(image_file):
	"""
	test the yellow_filter
	"""
	image = mpimg.imread(image_file)
	result = yellow_filter(image, low=(10,50,0), high=(30,255,255))

	plt.imshow(result, cmap='gray')
	plt.show()
	print(result.shape)

def test_white_filter(image_file):
	"""
	test the white_filter
	"""
	image = mpimg.imread(image_file)
	result = white_filter(image, low=(180,180,180), high=(255,255,255))

	plt.figure(),plt.imshow(image)
	plt.figure(),plt.imshow(result, cmap='gray')
	plt.show()
	print(result.shape)


if __name__ == "__main__":
	# these lib just use in test() functions
	import glob
	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt
	import pickle

	# test_draw()
	# test_thresh_image("../test_images/challenge/15.jpg", s_thresh=(50,150), sx_thresh=(20, 100))
	test_thresh_images("../output_images/", "../examples/", s_thresh=(50,150), sx_thresh=(20, 100))
	# test_yellow_filter("../output_images/challenge/undistort/1.jpg")
	# test_yellow_grid_thresh_images("../output_images/challenge/undistort/", "../output_images/challenge/threshed/", y_low=(10,50,0), y_high=(30,255,255), sx_thresh=(20, 100))
	# test_white_filter("../output_images/challenge/undistort/1.jpg")
	# test_yellow_white_thresh_images("../output_images/challenge/undistort/", "../output_images/challenge/threshed/", y_low=(10,50,0), y_high=(30,255,255), w_low=(190,190,190), w_high=(255,255,255))

	# test_yellow_white_thresh_images("../output_images/harder/undistort/", "../output_images/harder/threshed/", 
										# y_low=(10,50,0), y_high=(30,255,255), w_low=(190,190,190), w_high=(255,255,255))
	# test_y_w_dynamic_images("../output_images/harder/undistort/", "../output_images/harder/threshed/", 
								# y_low=(10,0,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255))
	# test_color_grid_thresh_dynamic("../output_images/harder/undistort/", "../output_images/harder/threshed/", 
								# s_thresh=(50,150), sx_thresh=(20, 100))