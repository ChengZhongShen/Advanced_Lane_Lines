# use straight_lines1.jpg to generate view_perspective transform parameter M & Minv

import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

def main():
	image = mpimg.imread("../output_images/undistort/straight_lines1.jpg")
	img_size = (image.shape[1], image.shape[0])

	# copied from lession instroduction and ajust manully
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

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

	# draw 4 lines
	print("draw lines on image...")
	cv2.line(image, tuple(src[0]), tuple(src[1]), [255,0,0], 2)
	cv2.line(image, tuple(src[1]), tuple(src[2]), [255,0,0], 2)
	cv2.line(image, tuple(src[2]), tuple(src[3]), [255,0,0], 2)
	cv2.line(image, tuple(src[3]), tuple(src[0]), [255,0,0], 2)

	cv2.line(warped, tuple(dst[0]), tuple(dst[1]), [255,0,0], 2)
	cv2.line(warped, tuple(dst[1]), tuple(dst[2]), [255,0,0], 2)
	cv2.line(warped, tuple(dst[2]), tuple(dst[3]), [255,0,0], 2)
	cv2.line(warped, tuple(dst[3]), tuple(dst[0]), [255,0,0], 2)

	# write to pickle file
	print("write to pickle file >>> trans_pickle.p")
	trans_pickle = {}
	trans_pickle["M"] = M
	trans_pickle["Minv"] = Minv 
	pickle.dump(trans_pickle, open("trans_pickle.p", "wb"))

	print("Visulize the result...")
	f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
	ax1.imshow(image), ax1.set_title('Undistored Image', fontsize=15)
	ax2.imshow(warped), ax2.set_title('Wraped Image', fontsize=15)
	plt.show()


# test if the tran is work
def test():
	pickle_file = open("trans_pickle.p", "rb")
	trans_pickle = pickle.load(pickle_file)
	M = trans_pickle["M"]  
	Minv = trans_pickle["Minv"]

	img_size = (1280, 720)

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

if __name__ == '__main__':
	main()
	# test()