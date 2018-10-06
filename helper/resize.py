# resize the image size 1280X720 to 640X360 for report show

import cv2

def resize(src, dst):
	img = cv2.imread(src)
	img_resize = cv2.resize(img, (640, 320))
	print("writing... >>", dst)
	cv2.imwrite(dst, img_resize)

if __name__ == "__main__":
	resize("../examples/test6_threshed_wraped.jpg", "../examples/test6_threshed_wraped_resize.jpg") 
	# resize("../output_images/threshed/test6.jpg", "../output_images/threshed/test6_resize.jpg") 
	# resize("../output_images/wraped/test6.jpg", "../output_images/wraped/test6_resize.jpg") 
	# resize("../output_images/test6.jpg", "../output_images/test6_resize.jpg") 
	# resize("../examples/project_detect_fail.png", "../examples/project_detect_fail_resize.png") 
	# resize("../examples/project_detect_fail_with_debug.png", "../examples/project_detect_fail_with_debug_resize.png") 
	# resize("../examples/576.jpg", "../examples/576_resize.jpg") 

