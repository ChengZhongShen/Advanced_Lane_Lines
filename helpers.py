import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

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




# # Define a class to receive the characteristics of each line detection
# class Line():
# 	def __init__(self):
# 		# was the line detected in the last iteration?
# 		self.detected = Flase
# 		# x values of the last n fits of the line
# 		self.recent_xfitted = []
# 		# average x values of the fitted line over the last n iterations
# 		self.bestx = None
# 		# polynomial coefficients for the most recent fit
# 		self.current_fit = [np.array([Flase])]
# 		# radius of curvature of the line in some units
# 		self.radius_of_curvature = None
# 		# distance in meters of vechicle center from the line
# 		self.line_base_pos = None
# 		# difference in fit coefficients between last and new fits
# 		self.diffs = np.array([0,0,0], dtype='float')
# 		# x values for detected line pixels
# 		self.allx = None
# 		# y values for detected line pixels
# 		self.ally = None


# # Drawing
# # Create an image to draw the lines on
# warp_zero = np.zeros_like(warped).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# # Recast the x and y points into usable format for cv2.fillPoly()
# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# pts = np.hstack((pts_left, pts_right))

# # Draw the lane onto the warped blank image
# cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

# # Warp the blank back to original image space using inverse perspective matrix(Minv)
# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# # Combine the result with the original image
# result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
# plt.imshow(result)