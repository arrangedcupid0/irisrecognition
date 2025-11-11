import cv2
import iris
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

iris_pipeline = iris.IRISPipeline()
image = "/home/arrangedcupid0/irisrecognition/users/user1/119.jpg"
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)	
output1 = iris_pipeline(iris.IRImage(img_data=img, image_id="test", eye_side="right"))
image = "/home/arrangedcupid0/irisrecognition/users/user1/122.jpg"
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)	
output2 = iris_pipeline(iris.IRImage(img_data=img, image_id="gen", eye_side="right"))
image = "/home/arrangedcupid0/irisrecognition/users/user2/76.jpg"
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)	
output3 = iris_pipeline(iris.IRImage(img_data=img, image_id="imp", eye_side="right"))


test = output1['iris_template']
gen = output2['iris_template']
imp = output3['iris_template']

matcher = iris.HammingDistanceMatcher()
#def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:

genScore = matcher.run(test, gen)
impScore = matcher.run(test, imp)

print(f"Genuine and Imposter Scores: {genScore}, {impScore}")
