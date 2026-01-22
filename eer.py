import cv2
import iris
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

values = 30
lower = .31
step = .16 / values

for user in os.listdir("/home/arrangedcupid0/irisrecognition/users"):
	print(user)
	if Path(f"/home/arrangedcupid0/irisrecognition/users/{user}/impArr.txt").exists():
		impArr = np.loadtxt(f"/home/arrangedcupid0/irisrecognition/users/{user}/impArr.txt", delimiter=",")
	else:
		print("no impArr")
		continue
	if Path(f"/home/arrangedcupid0/irisrecognition/users/{user}/genArr.txt").exists():
		genArr = np.loadtxt(f"/home/arrangedcupid0/irisrecognition/users/{user}/genArr.txt", delimiter=",")
	else:
		print("no genArr")
		continue

	farArr = []
	with open(f"/home/arrangedcupid0/irisrecognition/users/{user}/FAR.txt", 'w', encoding="utf-8") as FAR:
		FAR.write("FAR at threshold:\n")
		for i in range(values):
			threshold = (step * i) + lower
			FAR.write(str(round(threshold,2)) + ": ")
			farCount = 0
			for j in range(len(impArr)):
				if impArr[j] <= threshold:
					farCount += 1
			far = (farCount/len(impArr)) * 100
			farArr.append(far)
			FAR.write(str(far) + "%\n")
	frrArr = []
	with open(f"/home/arrangedcupid0/irisrecognition/users/{user}/FRR.txt", 'w', encoding="utf-8") as FRR:
		FRR.write("FRR at threshold:\n")
		for i in range(values):
			threshold = (step * i) + lower
			FRR.write(str(round(threshold,2)) + ": ")
			frrCount = 0
			for j in range(len(genArr)):
				if genArr[j] > threshold:
					frrCount += 1
			frr = (frrCount/len(genArr)) * 100
			frrArr.append(frr)
			FRR.write(str(frr) + "%\n")
	closest = 1
	closeInd = 0
	for i in range(values):
		dist = abs(farArr[i] - frrArr[i])
		if dist < closest:
			closeInd = i
			closest = dist
	print("At threshold " + str((step * closeInd) + lower) + ", FAR: " + str(farArr[closeInd]) + " and FRR: " + str(frrArr[closeInd]))
