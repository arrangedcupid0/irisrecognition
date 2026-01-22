import cv2
import iris
import os
import numpy as np

iris_pipeline = iris.IRISPipeline()
matcher = iris.HammingDistanceMatcher()

for user in os.listdir("/home/arrangedcupid0/irisrecognition/users"):
	print(f"user: {user}")
	sample = cv2.imread(f"/home/arrangedcupid0/irisrecognition/imposters/{each}", cv2.IMREAD_GRAYSCALE)
	output = iris_pipeline(iris.IRImage(img_data=sample, image_id="test", eye_side="right"))
	sampOut = output['iris_template']
	if sampOut is None:
		print("Bad read.")
		continue
	impArr = []
	iFailCount = 0
	for other in os.listdir("/home/arrangedcupid0/irisrecognition/imposters"):
		if each is not other:
			imp = cv2.imread(f"/home/arrangedcupid0/irisrecognition/imposters/{other}", cv2.IMREAD_GRAYSCALE)
			output = iris_pipeline(iris.IRImage(img_data=imp, image_id="test", eye_side="right"))
			impOut = output['iris_template']
			if impOut is None:
				iFailCount += 1
				continue
			impArr.append(matcher.run(sampOut, impOut))
	genArr = []
	gFailCount = 0
	for other in os.listdir(f"/home/arrangedcupid0/irisrecognition/users/{user}"):
		if other == "impArr.txt" or other == "genArr.txt":
			continue
		gen = cv2.imread(f"/home/arrangedcupid0/irisrecognition/users/{user}/{other}", cv2.IMREAD_GRAYSCALE)
		output = iris_pipeline(iris.IRImage(img_data=gen, image_id="test", eye_side="right"))
		genOut = output['iris_template']
		if genOut is None:
			gFailCount += 1
			continue
		genArr.append(matcher.run(sampOut, genOut))
	impArr = np.array(impArr)
	genArr = np.array(genArr)
	np.savetxt(f"/home/arrangedcupid0/irisrecognition/users/{user}/impArr.txt", impArr, delimiter=",")
	np.savetxt(f"/home/arrangedcupid0/irisrecognition/users/{user}/genArr.txt", genArr, delimiter=",")
	print(f"Failed imposter code count: {iFailCount}")
	print(f"Failed genuine code count: {gFailCount}")
