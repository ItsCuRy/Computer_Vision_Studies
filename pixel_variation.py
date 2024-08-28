import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread('Photos/danilo.jpg')

#transformacao para preto e branco 

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Preto e branco", img)

#inversao de imagem 
def image_invertion(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            image[y, x] = 255 - image[y, x]

    return image

inverted = image_invertion('Photos/danilo.jpg')
cv2.imshow("Reversa", inverted) 

#Histograma da imagem p&b

hist = cv2.calcHist([img], [0], None, [256], [0, 256]) 
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

#funcao de aumentar intensidade 

def intensit_changer(img):
    value = int(input("Escreva o quanto voce quer aumentar ou diminuir a intensidade dos pixels da imagem: "))
    # Obtenha as dimensões da imagem
    height, width = img.shape

    # Itere sobre cada pixel e adicione a intensidade
    for y in range(height):
        for x in range(width):
            new_value = img[y, x] + value
            img[y, x] = np.clip(new_value, 0, 255)
    
    return img


img_changed = intensit_changer(img)
cv2.imshow("Imaged Changed",img_changed)



cv2.waitKey(0)


