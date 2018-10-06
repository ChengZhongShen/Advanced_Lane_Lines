# handle the video use the pipeline
print("import lib...")
from moviepy.editor import VideoFileClip
import os
print("import pipeline...")
from pipeline import *

def gen_video(video, subclip=False):
	'''
	handle the project_video use pipeline function
	'''
	# check if handle the subclip
	if subclip:
		print("test on 5 second video")
		clip = VideoFileClip("test_video/"+video).subclip(0,5)
	else:
		print("handle the whole video")
		clip = VideoFileClip("test_video/"+video)
	
	white_output = "output_video/"+video
	white_clip = clip.fl_image(pipeline)
	white_clip.write_videofile(white_output, audio=False)

def gen_video_tracker(video, subclip=False, debug_window=False):
	'''
	handle the project_video use pipeline function
	'''
	# Create pipeline instance
	print("build pipeline instance...")
	left = Line()
	right = Line()
	pipeline = Pipeline(left, right)

	# checkif debug_window if turn on
	pipeline.debug_window = True if debug_window else False

	# check if handle the subclip
	if subclip:
		print("test on 5 second video")
		clip = VideoFileClip("test_video/"+video).subclip(0,5)
	else:
		print("handle the whole video")
		clip = VideoFileClip("test_video/"+video)
	
	# choice the pipeline according the video
	if video == "project_video.mp4":
		white_clip = clip.fl_image(pipeline.pipeline)
	elif video == "challenge_video.mp4":
		white_clip = clip.fl_image(pipeline.pipeline_challenge)
	elif video == "harder_challenge_video.mp4":
		white_clip = clip.fl_image(pipeline.pipeline_harder)
	else: 
		print("Wrong Vidoe Name, please check the video name!!!")
		return 1

	white_output = "output_video/temp/"+video
	white_clip.write_videofile(white_output, audio=False)

	print("processed", pipeline.image_counter, "images")
	print("Detected Failure: ", pipeline.fit_fail_counter)
	print("Search Failure: ", pipeline.search_fail_counter)

def get_image(video, dst, frame_list):
	'''
	get image from the video
	frame_list = [1, 3, 5, 7, 9]
	'''
	clip = VideoFileClip(video)

	for t in frame_list:
		imgpath = os.path.join(dst, '{}.jpg'.format(t))
		clip.save_frame(imgpath, t)

if __name__ == "__main__":
	# get_image("./test_video/challenge_video.mp4", "./test_images/challenge/", [i for i in range(1,16)])
	# get_image("./test_video/harder_challenge_video.mp4", "./test_images/harder/", [i for i in range(1,47)])

	# # gen project video
	# gen_video_tracker("project_video.mp4", subclip=True) 
	# gen_video_tracker("project_video.mp4", subclip=False)

	# gen_video_tracker("challenge_video.mp4", subclip=True) 
	# gen_video_tracker("challenge_video.mp4", subclip=False)
	
	# gen_video_tracker("harder_challenge_video.mp4", subclip=False, debug_window=True)
	gen_video_tracker("harder_challenge_video.mp4", subclip=False, debug_window=False)