import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def generate_random_rgb_values(size):
    rgb_values = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(size)]
    return rgb_values

def top_three_frequencies(img,min_distance=15):
    #Converter imagem para P&B assumindo que ela é RGB
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular o histograma
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()

    # Flatten o histograma e pegar os índices dos três valores mais frequentes
    hist = hist.flatten()
     # Ordenar os índices dos valores de intensidade por frequência (em ordem decrescente)
    sorted_indices = hist.argsort()[::-1]
    
    # Selecionar os três valores de intensidade com maior frequência respeitando a distância mínima
    selected_values = []
    for idx in sorted_indices:
        if all(abs(idx - val) >= min_distance for val in selected_values):
            selected_values.append(idx)
        if len(selected_values) == 3:
            break
    
    # Ordenar os valores selecionados em ordem crescente
    selected_values.sort()
    #print("Selected values : {}".format(selected_values))
    
    return selected_values


#def color_definer(pixel_value,thresholds):
#    for k, threshold in enumerate(thresholds):


    
    
def multi_threshold(img, thresholds):
    img_copy = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    colors = generate_random_rgb_values(len(thresholds) + 1)

    for i in range(height):
        for j in range(width):

            pixel_value = gray_img[i, j]
            for k, threshold in enumerate(thresholds):
               
                if pixel_value < threshold:
                    img_copy[i, j] = colors[k]
                    break
            else:
                img_copy[i, j] = colors[-1]

    return img_copy


img = cv2.imread("Photos/pikachu2ComBackground.webp")


thresholds = top_three_frequencies(img)
#print("Thresholds: {}".format(thresholds))
start_time = time.time()
thresh_img = multi_threshold(img, thresholds)
end_time = time.time()

print(end_time - start_time)
cv2.imshow("Thresh", thresh_img)
cv2.waitKey(0)