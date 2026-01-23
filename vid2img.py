#so this one takes everything in the AffectiVR directory
import ffmpeg
import os
import random
import shutil

vidPath = "/mnt/c/Users/mstoll3/Desktop/AffectiVR"

for user in os.listdir(vidPath):
	if not os.path.exists(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/"):
		os.mkdir(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/")
	for vid in os.listdir(vidPath + f"/{user}"):
		if vid == "007":
			break
		ffmpeg.input(vidPath + f"/{user}/{vid}/eye1.mp4").output(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/{vid}%d.jpg", vf="fps=2").run()
	#verify valid iris codes?
	random.seed()
	os.mkdir(f"/mnt/c/Users/mstoll3/Desktop/imposters/{user}/")
	for i in range(21):
		sample = random.choice(os.listdir(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/"))
		while sample in os.listdir(f"/mnt/c/Users/mstoll3/Desktop/imposters/{user}/"):
			print("collision")
			sample = random.choice(os.listdir(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/"))
		shutil.copy(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/{sample}", f"/mnt/c/Users/mstoll3/Desktop/imposters/{user}/{sample}")
