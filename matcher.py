import cv2
import iris
import os
import numpy as np

iris_pipe = iris.IRISPipeline()
id = 1
matcher = iris.HammingDistanceMatcher()
all = 0

for gen in os.listdir("/home/arrangedcupid0/irisrecognition/imposters"):
	tru = cv2.imread(f"/home/arrangedcupid0/irisrecognition/imposters/{gen}", cv2.IMREAD_GRAYSCALE)
	output = iris_pipe(iris.IRImage(img_data=tru, image_id="test", eye_side="right"))
	genOut = output['iris_template']
	FAR = 0
	for imp in os.listdir("/home/arrangedcupid0/irisrecognition/imposters"):
		if gen is imp:
			print("same")
		else:
			fals = cv2.imread(f"/home/arrangedcupid0/irisrecognition/imposters/{imp}", cv2.IMREAD_GRAYSCALE)
			output = iris_pipe(iris.IRImage(img_data=fals, image_id=id, eye_side="right"))
			id += 1
			impOut = output['iris_template']
			print(f"ran again {id}")
			score = matcher.run(genOut, impOut)
			if score > .37:
				FAR += 1
	all += FAR
print(f"total FAR: {all}")
