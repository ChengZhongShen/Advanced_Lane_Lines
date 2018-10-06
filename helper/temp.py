import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import cv2
import numpy as np 

image = mpimg.imread("../output_images/harder/undistort/1.jpg")
HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

height = image.shape[0]
width = image.shape[1]
y_offset = 50	# the offset add to mean brightness
w_offset = 50	# the offset add to mean brightness
HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
V = HSV[:,:,2]
bright_lb = int(np.mean(V[height//2:, :width//2]))	# use left bottom corner brightness as ref
bright_rb = int(np.mean(V[height//2:, width//2:]))	# use right bottom corner brightness as ref
info_str1 = "brightness left bottom is: {}".format(bright_lb)
info_str2 = "brightness right bottom is: {}".format(bright_rb)
print(info_str1)
print(info_str2)

sx_thresh=(20, 100)
sobelx = cv2.Sobel(V, cv2.CV_64F, 1, 0) # Take the derivateive in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

# plt.figure(),plt.imshow(image),plt.title("RGB")
# plt.figure(),plt.imshow(HSV[:,:,0],cmap='gray'),plt.title("H")
# plt.figure(),plt.imshow(HSV[:,:,1],cmap='gray'),plt.title("S")
plt.figure(),plt.imshow(HSV[:,:,2],cmap='gray'),plt.title("V")
plt.figure(),plt.imshow(scaled_sobel,cmap='gray'),plt.title("sobel")
plt.figure(),plt.imshow(sxbinary,cmap='gray'),plt.title("sobel_filtered")


plt.show()