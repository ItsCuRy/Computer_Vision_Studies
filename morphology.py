import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Photos/pikachu2ComBackground.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", gray)
thresh = cv2.Canny(gray, 200, 200)
cv2.imshow("thresh", thresh)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 1)
cv2.imshow("Eroded", erosion)
thresh2 = cv2.Canny(erosion, 200, 200)
cv2.imshow("thresh eroded", thresh2)
cv2.waitKey(0)
