import cv2
import iris
import numpy as np
import pycuda

img = cv2.imread('06.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

out = cv2.drawKeypoints(img, keypoints, None)
cv2.imshow("Keypoints", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

