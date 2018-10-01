# functions for caluate the lan radius im meters

import numpy as np 

# ym_per_pix = 30/720
# xm_per_pix = 3.7/700

Y_MAX = 720 # this is the image height
X_MAX = 1280 # this is the image width

QUADRATIC_COEFF = 3e-4 # this is for the test data genrate

def generate_data():
	'''
	Generates fake data to use for calculating lane curvature
	only for testing
	'''
	# Set random seed number
	# Comment this out if want to see result s on different
	np.random.seed(0)
	# Generate some fake data to repreent lane-line pixels
	ploty = np.linspace(0, Y_MAX-1, num=Y_MAX) # to cover y range
	quadratic_coeff = QUADRATIC_COEFF # arbitray quadratic coefficient
	
	# For each y postion generate random x postion with +/- 50
	# of the line base postion in each case (x=200 for left, x =900 for right)
	
	# Try to use in line function lambad to instead of list compression, not work, reason not found 2018/9/30 chengzhong.shen
	# line_fun = lambad y: (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
	# leftx = 200 + line_fun(ploty)
	# rightx = 900 + line_fun(ploty)

	leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
						for y in ploty])
	rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
						for y in ploty])

	leftx = leftx[::-1]	# revers to match top-to-bottom in y
	rightx = rightx[::-1]	# revers to match top-to-bottom in y

	return leftx, ploty, rightx, ploty

def measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700):
	'''
	Calcualtes the curvature of polynomial functions in meters and the offset from the lancenter
	'''
	# Transform pixel to meters
	leftx = leftx * xm_per_pix
	lefty = lefty * ym_per_pix
	rightx = rightx * xm_per_pix
	righty = righty * ym_per_pix

	# fit the polynomial
	left_fit_cr = np.polyfit(lefty, leftx, 2)
	right_fit_cr = np.polyfit(righty, rightx, 2)

	# Define y-value where we want radius of curvature
	# choose the maximum y-value
	y_eval = Y_MAX * ym_per_pix

	# Implement the caculation of R_curve
	# Caluate the radius R = (1+(2Ay+B)^2)^3/2 / (|2A|)
	radius_fun = lambda A, B, y: (1+(2*A*y+B)**2)**(3/2) / abs(2*A)

	left_curverad = radius_fun(left_fit_cr[0], left_fit_cr[1], y_eval)
	right_curverad = radius_fun(right_fit_cr[0], right_fit_cr[1], y_eval)

	return left_curverad, right_curverad

def measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700):
	'''
	calculate the the offest from lane center
	'''
	# HYPOTHESIS : the camera is mounted at the center of the car
	# the offset of the lane center from the center of the image is 
	# distance from the center of lane
	
	# Transform pixel to meters
	leftx = leftx * xm_per_pix
	lefty = lefty * ym_per_pix
	rightx = rightx * xm_per_pix
	righty = righty * ym_per_pix

	# fit the polynomial
	left_fit_cr = np.polyfit(lefty, leftx, 2)
	right_fit_cr = np.polyfit(righty, rightx, 2)

	# Define y-value where we want radius of curvature
	# choose the maximum y-value
	y_eval = Y_MAX * ym_per_pix

	left_point = np.poly1d(left_fit_cr)(y_eval)
	right_point = np.poly1d(right_fit_cr)(y_eval)

	lane_center = (left_point + right_point) / 2
	image_center = X_MAX * xm_per_pix / 2

	offset = lane_center - image_center

	return offset


def test():
	leftx, lefty, rightx, righty = generate_data()
	left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty)
	offset = measure_offset(leftx, lefty, rightx, righty)

	print(left_curverad, 'm', right_curverad, 'm')
	print("offset is: ", offset, "m")

if __name__ == '__main__':
	test()
	
# #######################################################################################
# # Calculate the radius of curvature in meters for both lane line
# # Should see values of 533.75 and 648.16 here, if using
# # the default `generate_data` function with given seed number
# # the offset should be 1280/2 - (200+900)/2 = 90 90*3.7/700=-0.4757(-0.4925)
# ########################################################################################