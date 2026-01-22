import cv2
import iris
import os
import numpy as np

iris_pipeline = iris.IRISPipeline()
baseimage = "/home/arrangedcupid0/irisrecognition/06.jpg"
img = cv2.imread(baseimage, cv2.IMREAD_GRAYSCALE)
output = iris_pipeline(iris.IRImage(img_data=img, image_id="test", eye_side="right"))
sampOut = output['iris_template']

#remember: DIS-similarity score
bestMatch = 1
id = 1
matcher = iris.HammingDistanceMatcher()
bestName = ""

for image in os.listdir("/home/arrangedcupid0/irisrecognition/users/user1"):
	if f"/home/arrangedcupid0/irisrecognition/{image}" is not baseimage:
		img = cv2.imread(f"/home/arrangedcupid0/irisrecognition/users/user1/{image}", cv2.IMREAD_GRAYSCALE)
		output = iris_pipeline(iris.IRImage(img_data=img, image_id=str(id), eye_side="right"))
		id += 1
		if output['error'] is None:
			testOut = output['iris_template']
			score = matcher.run(sampOut, testOut)
			if score < bestMatch:
				bestMatch = score
				bestName = image

print(f"Best match was {bestMatch}, for id {bestName}.")
