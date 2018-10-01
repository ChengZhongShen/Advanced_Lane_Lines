# handle the video use the pipeline

from moviepy.editor import VideoFileClip

from pipeline import *

def project_video(subclip=False):
	'''
	handle the project_video use pipeline function
	'''
	# check if handle the subclip
	if subclip:
		print("test on 5 second video")
		clip1 = VideoFileClip("test_video/project_video.mp4").subclip(0,5)
	else:
		print("handle the whole video")
		clip1 = VideoFileClip("test_video/project_video.mp4")
	
	white_output = "output_video/project_video.mp4"
	white_clip = clip1.fl_image(pipeline)
	white_clip.write_videofile(white_output, audio=False)

def gen_video(video, subclip=False):
	'''
	handle the project_video use pipeline function
	'''
	# check if handle the subclip
	if subclip:
		print("test on 5 second video")
		clip1 = VideoFileClip("test_video/"+video).subclip(0,5)
	else:
		print("handle the whole video")
		clip1 = VideoFileClip("test_video/"+video)
	
	white_output = "output_video/"+video
	white_clip = clip1.fl_image(pipeline)
	white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
	# project_video(subclip=True)
	# project_video(subclip=False)
	# gen_video("project_video.mp4", subclip=True)
	# gen_video("challenge_video.mp4", subclip=True)
	gen_video("harder_challenge_video.mp4", subclip=True)