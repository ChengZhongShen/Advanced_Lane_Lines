import numpy as np 

ym_per_pix = 30/720
xm_per_pix = 3.7/700

def generate_data(ym_per_pix, xm_per_pix):
	'''
	Generates fake data to use for calculating lane curvature
	'''
	# Set random seed number
	# Comment this out if want to see result s on different
	np.random.seed(0)
	# Generate some fake data to repreent lane-line pixels
	ploty = np.linspace(0, 719, num=720) # to cover y range
	quadratic_coeff = 3e-4 # arbitray quadratic coefficient
	# For each y postion generate random x postion with +/- 50
	# of the line base postion in each case (x=200 for left, x =900 for right)
	# line_fun = lambad y: (y**2)*quadratic_coeff + np.random.randint(-50, high=51)

	# leftx = 200 + line_fun(ploty)
	# rightx = 900 + line_fun(ploty)
	leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
						for y in ploty])
	rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
						for y in ploty])

	leftx = leftx[::-1]	# revers to match top-to-bottom in y
	rightx = rightx[::-1]	# revers to match top-to-bottom in y

	# Fit as econd order polynomial to pixel postion
	# change from pixel to meters
	ploty = ploty * ym_per_pix
	leftx = leftx * xm_per_pix
	rightx = rightx * xm_per_pix

	left_fit_cr = np.polyfit(ploty, leftx, 2)
	right_fit_cr = np.polyfit(ploty, rightx, 2)

	return ploty, left_fit_cr, right_fit_cr

def measure_curvature_real():
	'''
	Calcualtes the curvature of polynomial functions in meters
	'''
	# Define conversion in x and y from pixel spaces to meters
	ym_per_pix = 30/720
	xm_per_pix = 3.7/700

	ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
	print(left_fit_cr)
	print(right_fit_cr)

	# Define y-value where we want radius of curvature
	# choose the maximum y-value
	y_eval = np.max(ploty)
	print(y_eval)

	# Implement the caculation of R_curve
	# Caluate the radius R = (1+(2Ay+B)^2)^3/2 / (|2A|)
	radius_fun = lambda A, B, y: (1+(2*A*y+B)**2)**(3/2) / abs(2*A)

	left_curverad = radius_fun(left_fit_cr[0], left_fit_cr[1], y_eval)
	right_curverad = radius_fun(right_fit_cr[0], right_fit_cr[1], y_eval)

	return left_curverad, right_curverad

# Calculate the radius of curvature in meters for both lane line
left_curverad, right_curverad = measure_curvature_real()

print(left_curverad, 'm', right_curverad, 'm')
# Should see values of 533.75 and 648.16 here, if using
# the default `generate_data` function with given seed number