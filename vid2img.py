#so this one takes everything in the AffectiVR directory
import ffmpeg
import os
import random

vidPath = "/mnt/c/Users/mstoll3/Desktop/AffectiVR"

for user in os.listdir(vidPath):
	if not os.path.exists(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/"):
		os.mkdir(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/")
	for vid in os.listdir(vidPath + f"/{user}"):
		if vid == "007":
			break
		ffmpeg.input(vidPath + f"/{user}/{vid}/eye1.mp4").output(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/{vid}%d.jpg", vf="fps=1").run()
	#random image select/copy?
