import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob

def find_lane_pixels(binary_warped):
	"""
	find lane in a binary_warped image
	input: binary_warped image
	output: left/right lane pixel poistion and a drawed search image
	"""

	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

	# plt.plot(histogram)
	# plt.show()

	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

	# Find the peak of the left and right havleve of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2) # 1280/2=640
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nWindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimu number of pixels found to recenter window
	min_pix = 50
	# Set height of windows - based on nWindows above adn image shape
	window_height = np.int(binary_warped.shape[0]//nWindows)
	# Identify the x and y positions of all nonzero(i.e. activated) pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0]) # y is row, x is col
	nonzerox = np.array(nonzero[1])
	# Current postions to be updated later for each window in n_window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nWindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height

		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visulization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low),
			(win_xleft_high, win_y_high), (0,255,0),2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low),
			(win_xright_high, win_y_high), (0,255,0),2)
		
		# plt.imshow(out_img)
		# plt.show()

		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple

		# Append these indices to hte lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# # update the window center for next round window scan
		if len(good_left_inds) > min_pix:
			leftx_current = int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > min_pix:
			rightx_current = int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of list)
	try:
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
		# Avoids an error if the above is not implemented fully
		pass

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# print(len(nonzerox))
	return (leftx, lefty, rightx, righty, out_img)

def find_lane_pixels_v2(binary_warped):
	"""
	find lane in a binary_warped image
	input: binary_warped image
	output: left/right lane pixel poistion and a drawed search image
	2018/10/4 update to v2, add
		1. search window side check, if the search window arrive the picture side/picture middle, stop search
		2. search window start piostion check, if one search window's start posiont is no reasonale, adjust acording another one
		3. noise window discard, if the is lot of finding, distcard, noise.
		4. blank window distcard, if two window is blank, stop search.
		5. plan, change the window size(when the lane is curved much, 
			the window counln't shift quickly, plan to change the window height half and just search half of the image. )
	"""
	img_height = binary_warped.shape[0]
	img_width = binary_warped.shape[1]


	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[img_height//2:,:], axis=0)

	# Create an output image to draw on and visualize the result
	# plt.figure(),plt.imshow(binary_warped, cmap='gray')
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# plt.figure(),plt.imshow(out_img, cmap='gray')
	# plt.show()
	# Find the peak of the left and right havleve of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2) # 1280/2=640
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# check if the left_base/right base is in range
	leftx_base_range = range(img_width//4-100, img_width//4+200)	# the right limit + 100 ()
	rightx_base_range = range(img_width//4*3-100, img_width//4+200)
	
	if (leftx_base in leftx_base_range) and (rightx_base in rightx_base_range): # good condistion
		pass
	elif ((leftx_base in leftx_base_range) == False) and ((rightx_base in rightx_base_range) == True):
		leftx_base = rightx_base + img_width//2
	elif ((leftx_base in leftx_base_range) == True) and ((rightx_base in rightx_base_range) == False):
		 rightx_base = leftx_base + img_width//2
	else: # if left and right start piont all not in range, set it to throyt piont forcely
		leftx_base = img_width//4
		rightx_base = img_width//4*3

	# print("leftx_base: ", leftx_base)
	# print("rightx_base: ", rightx_base)
	# plt.plot(histogram)
	# plt.show()

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nWindows = 18	# version 1 is 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimu number of pixels found to recenter window
	min_pix = 50 # 50 
	# Set height of windows - based on nWindows above adn image shape
	window_height = np.int(binary_warped.shape[0]//nWindows)
	# Identify the x and y positions of all nonzero(i.e. activated) pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0]) # y is row, x is col
	nonzerox = np.array(nonzero[1])
	# Current postions to be updated later for each window in n_window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# flag's
	l_stop = False	# if this flag is true, skip the search
	r_stop = False

	l_blank = False	# if the search result of this window is blank and the flag is True, set the stop flag true
	r_blank = False

	for window in range(nWindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height

		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# check the window postion
		if win_xleft_low < 0 or win_xleft_high > img_width//2:
			l_stop = True
			r_stop = True # when one side hit border, stop two side search
		if win_xright_low < img_width//2 or win_xright_high > img_width:
			r_stop = True
			l_stop = True # when one side hit border, stop two side search
		
		# plt.imshow(out_img)
		# plt.show()

		# check the stop flag the dothe search
		if not l_stop:
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
				(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple
			# Draw the windows on the visulization image
			cv2.rectangle(out_img, (win_xleft_low, win_y_low),
							(win_xleft_high, win_y_high), (0,255,0),2)
		if not r_stop: 
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
				(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple
			# Draw the windows on the visulization image
			cv2.rectangle(out_img, (win_xright_low, win_y_low),
							(win_xright_high, win_y_high), (0,255,0),2)
		
		# adjust this according thresh method, left 0.99, use color select, right use edagdetection
		noise_thresh_left = int(window_height*(margin+margin)*0.99) 
		noise_thresh_right = int(window_height*(margin+margin)*0.5)
		# # update the window center for next round window scan and searched list, only the pionts number meet the range, else is not searched, window is blank
		if (len(good_left_inds) > min_pix) and (len(good_left_inds) < noise_thresh_left):
			leftx_current = int(np.mean(nonzerox[good_left_inds]))
			left_lane_inds.append(good_left_inds)
		else:	# when there is no siganl < 50 or 30% window is ocupied, noise, consider this window is blank
			if l_blank == True: # secong blank window, stop search
				l_stop = True
			else:
				l_blank = True # firstime, set l_blank to True
		
		if (len(good_right_inds) > min_pix) and (len(good_right_inds) < noise_thresh_right):
			rightx_current = int(np.mean(nonzerox[good_right_inds]))
			right_lane_inds.append(good_right_inds)
		else:
			if r_blank == True: # second time blank window, stop search
				r_stop = True
			else:
				r_blank = True 	# firstime, set r_blank flag to True


	# print(len(left_lane_inds))
	# print(len(right_lane_inds))
	# Concatenate the arrays of indices (previously was a list of list)
	try:
		left_lane_inds = np.concatenate(left_lane_inds)
	except ValueError:
		# Avoids an error if the above is not implemented fully
		pass
	try:
		right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
		# Avoids an error if the above is not implemented fully
		pass

	# print(len(left_lane_inds))
	# print(len(right_lane_inds))
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# print(len(nonzerox))
	return (leftx, lefty, rightx, righty, out_img)

def fit_polynomial(leftx, lefty, rightx, righty, out_img):
	"""
	fit left and right lane polynomi
	"""
	# check if there is search failure
	if leftx.size == 0 or lefty.size == 0:
		cv2.putText(out_img,"Search failure", (50,60), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
		return out_img 

	# line fit use np.ployfit, second order, note lefty is x, leftx is y, later use ploty to get plotx
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# print(left_fit)
	# print(right_fit)

	left_lane_fun = np.poly1d(left_fit)
	right_lane_fun = np.poly1d(right_fit)

	# Generate x and y values for plotting
	ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
	left_plotx = left_lane_fun(ploty)
	right_plotx = right_lane_fun(ploty)

	# Visualization
	# Colors in the left and right lane regions
	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0 ,0 , 255]

	# draw fit line(sue 9 stright line)
	for i in range(0, 9):
		cv2.line(out_img, (int(left_plotx[i*79]), int(ploty[i*79])), (int(left_plotx[(i+1)*79]), int(ploty[(i+1)*79])), (255,255,0),2)
		cv2.line(out_img, (int(right_plotx[i*79]), int(ploty[i*79])), (int(right_plotx[(i+1)*79]), int(ploty[(i+1)*79])), (255,255,0),2)

	# Plots the left and right polynomials on the lane lines
	# plt.plot(left_plotx, ploty, color='yellow')
	# plt.plot(right_plotx, ploty, color='yellow')

	# print(len(leftx))
	# print(len(ploty))
	return out_img

def get_polynomial(leftx, lefty, rightx, righty, img_size):


	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	left_lane_fun = np.poly1d(left_fit)
	right_lane_fun = np.poly1d(right_fit)

	ploty = ploty = np.linspace(0, img_size[0]-1, img_size[0])
	left_fitx = left_lane_fun(ploty)
	right_fitx = right_lane_fun(ploty)

	return left_fitx, right_fitx, ploty

def lane_sanity_check(left_fitx, right_fitx, ploty):
	'''
	1. checking that they have similar curvature margin 10%
	2. checking that they are separated by approximately right distance horizontally
	tranform calibration distence 1280/2 margin 640, 5%(610-670) is good search, 15%(545-730) is detected
	3. Checking that they are roughly parallel, check the another side if 1280/4 margin 10%
	'''
	flag = True
	lane_distance_bot = right_fitx[720] - left_fitx[720]
	lane_distance_mid = right_fitx[320] - left_fitx[320]
	lane_distance_top = right_fitx[0] - left_fitx[0]
	
	# tranform calibration distence 1280/2 is 640, 5%(610-670) is good search, 15%(545-730) is detected
	if ((lane_distance_bot < 545) or (lane_distance_bot > 730)): flag = False
	if ((lane_distance_mid < 545) or (lane_distance_mid > 730)): flag = False
	if ((lane_distance_top < 500) or (lane_distance_top > 730)): flag = False # change top to 500, in some frame, the road in not flat, the lane will be small far from camera

	return flag, lane_distance_bot, lane_distance_mid, lane_distance_top

def lane_sanity_check_challenge(left_fitx, right_fitx, ploty):
	'''
	1. checking that they have similar curvature margin 10%
	2. checking that they are separated by approximately right distance horizontally
	tranform calibration distence 1280/2 margin 640, 5%(610-670) is good search, 15%(545-730) is detected
	3. Checking that they are roughly parallel, check the another side if 1280/4 margin 10%
	mannully adjust the threshold bot(480-600), mid(350-500), top(100-500)
	'''
	flag = True
	lane_distance_bot = right_fitx[720] - left_fitx[720]
	lane_distance_mid = right_fitx[320] - left_fitx[320]
	lane_distance_top = right_fitx[0] - left_fitx[0]
	
	# tranform calibration distence 1280/2 is 640, 5%(610-670) is good search, 15%(545-730) is detected
	if ((lane_distance_bot < 480) or (lane_distance_bot > 600)): flag = False
	if ((lane_distance_mid < 350) or (lane_distance_mid > 500)): flag = False
	if ((lane_distance_top < 150) or (lane_distance_top > 500)): flag = False

	return flag, lane_distance_bot, lane_distance_mid, lane_distance_top

def lane_sanity_check_harder(left_fitx, right_fitx, ploty):
	'''
	1. checking that they have similar curvature margin 10%
	2. checking that they are separated by approximately right distance horizontally
	tranform calibration distence 1280/2 margin 640, 5%(610-670) is good search, 15%(545-730) is detected
	3. Checking that they are roughly parallel, check the another side if 1280/4 margin 10%
	mannully adjust the threshold bot(480-600), mid(350-500), top(100-500)
	2018/10/4, need further change, return True firstly bot(400-700), mid(400-700), top(200-800)
														bot(400-700), mid(400-1000), top(200-2000)
														bot(400-700), mid(400-1200), top(200-2200)

	'''
	flag = True
	lane_distance_bot = right_fitx[720] - left_fitx[720]
	lane_distance_mid = right_fitx[320] - left_fitx[320]
	lane_distance_top = right_fitx[0] - left_fitx[0]
	
	# tranform calibration distence 1280/2 is 640, 5%(610-670) is good search, 15%(545-730) is detected
	if ((lane_distance_bot < 400) or (lane_distance_bot > 700)): flag = False
	if ((lane_distance_mid < 400) or (lane_distance_mid > 1200)): flag = False
	if ((lane_distance_top < 200) or (lane_distance_top > 2200)): flag = False

	return (flag, lane_distance_bot, lane_distance_mid, lane_distance_top)


###############################################################################
def test():
	# read the image and change to binary(when write binary to RGB *255)
	binary_warped = mpimg.imread('../output_images/wraped/test6.jpg')
	binary_warped = binary_warped[:,:,0]# get 1 channel, three channel is same

	out_img = fit_polynomial(binary_warped)

	plt.imshow(out_img)
	plt.show()

def test(binary_image_file, search="v1"):
	# read the image and change to binary(when write binary to RGB *255)
	binary_warped = mpimg.imread(binary_image_file)
	binary_warped = binary_warped[:,:,0] / 255 # get 1 channel, three channel is same
	plt.figure(),plt.imshow(binary_warped, cmap='gray'),plt.show()
	from cal_curv import measure_curv
	img_size = (binary_warped.shape[1], binary_warped.shape[0])
	
	# check the search flag to decide use which search function
	if search == "v1":
		leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
	elif search == "v2":
		leftx, lefty, rightx, righty, out_img = find_lane_pixels_v2(binary_warped)
	else:
		print("Search file is not correct!!")
		return 1
	# plt.figure(),plt.imshow(out_img)

	if len(leftx)==0 or len(rightx) == 0:
		print("Search Failure")
		return 

	left_fitx, right_fitx, ploty = get_polynomial(leftx, lefty, rightx, righty, img_size)
	left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
	flag, lane_distance_bot, lane_distance_mid, lane_distance_top = lane_sanity_check(left_fitx, right_fitx, ploty)

	cur_left = "left: {}".format(int(left_curverad))
	cur_right = "right: {}".format(int(right_curverad))
	info_str = "{}, {}, {}, {}".format(flag, int(lane_distance_bot), int(lane_distance_mid), int(lane_distance_top))
	out_img = fit_polynomial(leftx, lefty, rightx, righty, out_img)
	cv2.putText(out_img,cur_left, (50,580), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
	cv2.putText(out_img,cur_right, (50,640), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
	cv2.putText(out_img,info_str, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
	


	plt.figure(),plt.imshow(out_img)
	plt.show()

if __name__ == '__main__':
	# test('../output_images/project/wraped/test6.jpg')
	test('../examples/test6_threshed_wraped.jpg', search="v1")