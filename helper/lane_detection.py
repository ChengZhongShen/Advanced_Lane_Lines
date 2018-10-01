import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

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
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))

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
	return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
	"""
	fit left and right lane polynomi
	"""

	# Get the left/right lane pix point throught window shift search.
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

	# line fit use np.ployfit, second order, note lefty is x, leftx is y, later use ploty to get plotx
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# print(left_fit)
	# print(right_fit)

	left_lane_fun = np.poly1d(left_fit)
	right_lane_fun = np.poly1d(right_fit)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_plotx = left_lane_fun(ploty)
	right_plotx = right_lane_fun(ploty)

	# Visualization
	# Colors in the left and right lane regions
	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0 ,0 , 255]

	# Plots the left and right polynomials on the lane lines
	plt.plot(left_plotx, ploty, color='yellow')
	plt.plot(right_plotx, ploty, color='yellow')

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


###############################################################################
def test():
	# read the image and change to binary(when write binary to RGB *255)
	binary_warped = mpimg.imread('../output_images/wraped/test6.jpg')
	binary_warped = binary_warped[:,:,0] # get 1 channel, three channel is same

	out_img = fit_polynomial(binary_warped)

	plt.imshow(out_img)
	plt.show()


if __name__ == '__main__':
	test()