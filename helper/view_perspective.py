# use straight_lines1.jpg to generate view_perspective transform parameter M & Minv

import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

image = mpimg.imread("../output_images/undistort/straight_lines1.jpg")
img_size = (image.shape[1], image.shape[0])

# copied from lession instroduction.
src = np.float32(
    [[(img_size[0] / 2) - 63, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

# print(src.shape)
# exit()

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

def draw_lines(img, points, color=[255,0,0], thickness=2):
	"""
	draw 4 lines in image
	"""
	cv2.line(img, tuple(points[0]), tuple(points[1]), color, thickness)
	cv2.line(img, tuple(points[1]), tuple(points[2]), color, thickness)
	cv2.line(img, tuple(points[2]), tuple(points[3]), color, thickness)
	cv2.line(img, tuple(points[3]), tuple(points[0]), color, thickness)

def show_images():
	draw_lines(image, src)
	draw_lines(warped, dst)

	plt.figure()
	plt.imshow(image)

	plt.figure()
	plt.imshow(warped)

	plt.show()

# show_images()

# write to pickle file
def write_to_pickle():
	trans_pickle = {}
	trans_pickle["M"] = M
	trans_pickle["Minv"] = Minv 
	pickle.dump(trans_pickle, open("trans_pickle.p", "wb"))


# check the trans
def check_trans():
	pickle_file = open("trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]  
	Minv = trans_pickle["Minv"]

	image_files = glob.glob("../output_images/undistort/*.jpg")
	for idx, file in enumerate(image_files):
		print(file)
		img = mpimg.imread(file)
		warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
		file_name = file.split("\\")[-1]
		print(file_name)
		out_image = "../output_images/perspect_trans/"+file_name
		print(out_image)
		# convert to opencv BGR format
		warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_image, warped)

# check_trans()