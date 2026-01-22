import cv2
import iris
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

iris_pipeline = iris.IRISPipeline()
id = 0
with open('iris_templates.csv', 'w', newline='') as csvFile:
	writer = csv.writer(csvFile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
#for folder in os.listdir("/users/"):
#	for image in os.listdir(folder):
	image = "06.jpg"
	img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

#iris_visualizer = iris.visualisation.IRISVisualizer()
#canvas = iris_visualizer.plot_ir_image(iris.IRImage(img_data=img, eye_side="right"))
#plt.show()
	output = iris_pipeline(iris.IRImage(img_data=img, image_id = str(id), eye_side="right"))
#canvas = iris_visualizer.plot_iris_template(output["iris_template"])
#plt.show()
	id += 1
	if output["error"] is None:
		
		#np.savetxt("iris_templates.csv", output["iris_template"], fmt='%r', delimiter=",")
	else:
		print(f"error occurred in image number {id}.\n")
