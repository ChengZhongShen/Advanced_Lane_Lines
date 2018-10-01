# File include the image process functions

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


######################################
def test():
	import matplotlib.pyplot as plt 
	import matplotlib.image as mpimg
	import pickle

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


if __name__ == "__main__":
	test()