import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def gerar_valores_rgb_aleatorios(tamanho):
    valores_rgb = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(tamanho)]
    return valores_rgb

def multi_threshold(img, thresholds):
    img_copy = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    altura, largura = gray_img.shape
    cores = gerar_valores_rgb_aleatorios(len(thresholds) + 1)
    

    for i in range(altura):
        for j in range(largura):
            
            if gray_img[i, j] < thresholds[0]:
                img_copy[i, j] = cores[0]

            elif gray_img[i, j] >= thresholds[-1]:
                img_copy[i, j] = cores[-1]

            for k in range(1, len(thresholds)):
                if gray_img[i, j] >= thresholds[k-1] and gray_img[i, j] < thresholds[k]:
                    img_copy[i, j] = cores[k]

    return img_copy   


img = cv2.imread("Photos/pikachu2ComBackground.webp")
start = time.time()
thresh_img = multi_threshold(img, [50,100,150,200])
end = time.time()

cv2.imshow("Thresh", thresh_img)
print(end - start)
cv2.waitKey(0)