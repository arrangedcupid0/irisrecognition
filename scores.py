import cv2
import iris
import os
import numpy as np
import random

iris_pipeline = iris.IRISPipeline()
matcher = iris.HammingDistanceMatcher()
random.seed()

for user in os.listdir("/mnt/c/Users/mstoll3/Desktop/imposters/"):
	print(f"user: {user}")
	sampOut = None
	while sampOut is None:
		test = random.choice(os.listdir(f"/mnt/c/Users/mstoll3/Desktop/imposters/{user}"))
		sample = cv2.imread(f"/mnt/c/Users/mstoll3/Desktop/imposters/{user}/{test}", cv2.IMREAD_GRAYSCALE)
		output = iris_pipeline(iris.IRImage(img_data=sample, image_id="test", eye_side="right"))
		sampOut = output['iris_template']
	impArr = []
	iFailCount = 0
	for other in os.listdir("/mnt/c/Users/mstoll3/Desktop/imposters"):
		if other is not user:
			for image in os.listdir(f"/mnt/c/Users/mstoll3/Desktop/imposters/{other}"):
				imp = cv2.imread(f"/mnt/c/Users/mstoll3/Desktop/imposters/{other}/{image}", cv2.IMREAD_GRAYSCALE)
				output = iris_pipeline(iris.IRImage(img_data=imp, image_id="test", eye_side="right"))
				impOut = output['iris_template']
				if impOut is None:
					iFailCount += 1
					continue
				impArr.append(matcher.run(sampOut, impOut))
	genArr = []
	gFailCount = 0
	for other in os.listdir(f"/mnt/c/Users/mstoll3/Desktop/users/{user}"):
		if other == "impArr.txt" or other == "genArr.txt":
			continue
		gen = cv2.imread(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/{other}", cv2.IMREAD_GRAYSCALE)
		output = iris_pipeline(iris.IRImage(img_data=gen, image_id="test", eye_side="right"))
		genOut = output['iris_template']
		if genOut is None:
			gFailCount += 1
			continue
		genArr.append(matcher.run(sampOut, genOut))
	impArr = np.array(impArr)
	genArr = np.array(genArr)
	np.savetxt(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/impArr.txt", impArr, delimiter=",")
	np.savetxt(f"/mnt/c/Users/mstoll3/Desktop/users/{user}/genArr.txt", genArr, delimiter=",")
	print(f"Failed imposter code count: {iFailCount}")
	print(f"Failed genuine code count: {gFailCount}")
