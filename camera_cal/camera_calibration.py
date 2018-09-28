# python 3.x

import numpy as np 
import cv2
import glob
import matplotlib.pyplot as plt 

# !!! IMPORTANT, set the nx, ny according the calibration chessboard pictures.
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d pionts in image plane.

# Make a list of calibration images
images = glob.glob('chessboard_img/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw and display the corners
# 		cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
# 		write_name = 'corners_found'+str(idx)+'.jpg'
# 		cv2.imwrite(write_name, img)
# 		cv2.imshow('img', img)
# 		cv2.waitKey(500)
# cv2.destroyAllWindows()

import pickle

# Test undistorion on an image
img = cv2.imread('chessboard_img/calibration1.jpg')
img_size = (img.shape[1],img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('test_undist.jpg', dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist 
pickle.dump(dist_pickle, open("camera_cal.p", "wb"))
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# Visualize undistortion
f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistored Image', fontsize=30)

plt.show()
