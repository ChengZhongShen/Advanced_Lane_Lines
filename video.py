# handle the video use the pipeline

from moviepy.editor import VideoFileClip

from pipeline import *

def project_video(subclip=False):
	white_output = "output_video/project_video.mp4"
	if subclip:
		print("test on 5 second video")
		clip1 = VideoFileClip("test_video/project_video.mp4").subclip(0,5)
	else:
		print("handle the whole video")
		clip1 = VideoFileClip("test_videos/project_video.mp4")
	white_clip = clip1.fl_image(pipeline)
	white_clip.write_videofile(white_output, audio=False)

project_video(subclip=True)