# handle the video use the pipeline

from moviepy.editor import VideoFileClip

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

def gen_video_tracker(video, subclip=False):
	'''
	handle the project_video use pipeline function
	'''
	# Create pipeline instance
	left = Line()
	right = Line()
	pipeline = Pipeline(left, right)

	# check if handle the subclip
	if subclip:
		print("test on 5 second video")
		clip = VideoFileClip("test_video/"+video).subclip(0,5)
	else:
		print("handle the whole video")
		clip = VideoFileClip("test_video/"+video)
	
	white_output = "output_video/"+video
	white_clip = clip.fl_image(pipeline.pipeline)
	white_clip.write_videofile(white_output, audio=False)

	print("processed", pipeline.image_counter, "images")
	print("Detected Failure: ", pipeline.detected_fail_counter)

if __name__ == "__main__":
	# gen_video("project_video.mp4", subclip=True)
	# gen_video("challenge_video.mp4", subclip=True)
	# gen_video("harder_challenge_video.mp4", subclip=False)
	# gen_video("challenge_video.mp4", subclip=False)
	# gen_video_tracker("project_video.mp4", subclip=True) 
	# gen_video_tracker("project_video.mp4", subclip=False)
	gen_video_tracker("challenge_video.mp4", subclip=False) 
	# gen_video_tracker("harder_challenge_video.mp4", subclip=False)  
