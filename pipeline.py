# the image process pipeline for the lane detection.

# import the libary
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob
# import the function
from helper.helpers import get_camera_cal, get_perspective_trans
from helper.image_process import color_grid_thresh, draw_lane_fit, draw_lane_find
from helper.lane_detection import find_lane_pixels, get_polynomial, fit_polynomial, lane_sanity_check
from helper.cal_curv import measure_curv, measure_offset
from helper.lane_tracker import Line

# Get cal and tranform parameter
mtx, dist = get_camera_cal()
M, Minv = get_perspective_trans()

# create the left, right tracker and image counter
left = Line()
right = Line()
image_counter = 0

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
	result = draw_lane_find(image_undist, image_warped, Minv, leftx, lefty, rightx, righty)
	result = draw_lane_fit(result, image_warped, Minv, left_fitx, right_fitx, ploty)

	# write curverad and offset on to result image
	direction = "right" if offset < 0 else "left"
	str_cur = "Radius of Curvature = {0:.2f}(m)".format(curverad)
	str_offset = "Vehicle is {0:.2f}m ".format(offset) + "{} of center".format(direction)
	cv2.putText(result, str_cur, (50,60), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
	cv2.putText(result, str_offset, (50,120), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)

	return result


##################################################
def one_image_test():
	'''
	test the pipeline in one picture and show the result
	'''
	image = mpimg.imread("test_images/test6.jpg")

	result = pipeline(image)

	f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
	ax1.imshow(image), ax1.set_title('Original Image', fontsize=15)
	ax2.imshow(result), ax2.set_title('Processed Image', fontsize=15)
	plt.show()

def images_test(src, dst):
	'''
	test the pipeline on src folder's images 
	write the result to the dst folder
	'''
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


# class to keep left/right line
class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		# average x values of the fitted line over the last n iterations
		self.bestx = None
		# polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		# radius of curvature of the line in some units
		self.radius_of_curvature = None
		# distance in meters of vechicle center from the line
		self.line_base_pos = None
		# difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')
		# x values for detected line pixels
		self.allx = []
		# y values for detected line pixels
		self.ally = []


class Pipeline():
	'''
	pipeline class with has left/right line class to hold the related information
	'''
	def __init__(self, left, right):
		'''
		initial with left, right line class
		'''
		# the left line
		self.left = left
		# the right line
		self.right = right
		# the image counter
		self.image_counter = 0
		self.detected_fail_counter = 0
	
	def pipeline(self, image):
		# counter the image
		self.image_counter += 1
		
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

		# get the polynomial line points
		left_fitx, right_fitx, ploty = get_polynomial(leftx, lefty, rightx, righty, img_size)

		# sanity Check
		detected, lane_distance_bot, lane_distance_mid, lane_distance_top = lane_sanity_check(left_fitx, right_fitx, ploty)
		
		if detected:

			# store the left_fitx, right_fitx
			self.left.recent_xfitted.append(left_fitx)
			self.right.recent_xfitted.append(right_fitx)
			# store the leftx, lefty, rightx, righty
			self.left.allx.append(leftx)
			self.left.ally.append(lefty)
			self.right.allx.append(rightx)
			self.right.ally.append(righty)


		else:	# use previous data
			self.detected_fail_counter += 1
			# use the recent_xfitted
			left_fitx = self.left.recent_xfitted[-1]
			right_fitx = self.right.recent_xfitted[-1]
			# use the recent pionts
			leftx = self.left.allx[-1]
			lefty = self.left.ally[-1]
			rightx = self.right.allx[-1]
			righty = self.right.ally[-1]


		# measure the curverad and offset
		left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
		curverad = (left_curverad + right_curverad) / 2
		offset = measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)

		# draw the lane to the undist image
		if detected:	# only draw the detected pionts
			result = draw_lane_find(image_undist, image_warped, Minv, leftx, lefty, rightx, righty)
			result = draw_lane_fit(result, image_warped, Minv, left_fitx, right_fitx, ploty)
		else:
			result = draw_lane_fit(image_undist, image_warped, Minv, left_fitx, right_fitx, ploty)

		# write curverad and offset on to result image
		direction = "right" if offset < 0 else "left"
		str_cur = "Radius of Curvature = {}(m)".format(int(curverad))
		str_offset = "Vehicle is {0:.2f}m ".format(offset) + "{} of center".format(direction)
		cv2.putText(result, str_cur, (50,60), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
		cv2.putText(result, str_offset, (50,120), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)

		# #########################################################
		# # the right-up corner window for debug.
		# fit_image = fit_polynomial(image_warped)
		
		# # santiy check window			
		# cur_left = "left: {}".format(int(left_curverad))
		# cur_right = "right: {}".format(int(right_curverad))
		# info_str = "{}, {}, {}, {}".format(detected, int(lane_distance_bot), int(lane_distance_mid), int(lane_distance_top))

		# cv2.putText(fit_image,cur_left, (50,580), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
		# cv2.putText(fit_image,cur_right, (50,640), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
		# cv2.putText(fit_image,info_str, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
		
		# # set the window
		# fit_image_resize = cv2.resize(fit_image, (640, 360))
		# result[:360,640:]=fit_image_resize

		# # write the failure image for ananlysis
		# if not detected:
		# 	fileName = "./temp/"+str(self.image_counter)+".jpg"
		# 	write_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
		# 	cv2.imwrite(fileName, write_img)
		# ##############################################################
		
		return result

def one_image_test_tracker():
	'''
	test the pipeline in one picture and show the result
	'''
	image = mpimg.imread("test_images/test6.jpg")

	left = Line()
	right = Line()
	pipeline = Pipeline(left, right)
	result = pipeline.pipeline(image)

	print("processed", pipeline.image_counter, "images")

	plt.imshow(result)
	plt.show()

	# f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
	# ax1.imshow(image), ax1.set_title('Original Image', fontsize=15)
	# ax2.imshow(result), ax2.set_title('Processed Image', fontsize=15)
	# plt.show()

def images_test_tracker(src, dst):
	'''
	test the pipeline on src folder's images 
	write the result to the dst folder
	'''
	# create pipeline instance
	left = Line()
	right = Line()
	pipeline = Pipeline(left, right)

	image_files = glob.glob(src+"*.jpg")
	for idx, file in enumerate(image_files):
		print("handle on: ", file)
		img = mpimg.imread(file)
		result = pipeline.pipeline(img)
		file_name = file.split("\\")[-1]
		# print(file_name)
		out_image = dst+file_name
		# print(out_image)
		image_dist = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_image, image_dist)

	print("processed", pipeline.image_counter, "images")
	print("Detected Failure: ", pipeline.detected_fail_counter)


if __name__ == '__main__':
	# one_image_test()
	# images_test("test_images/", "output_images/")
	# one_image_test_tracker()
	images_test_tracker("test_images/", "output_images/")