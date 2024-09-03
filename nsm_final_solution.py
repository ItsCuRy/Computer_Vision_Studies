import cv2
import numpy as np
from scipy.interpolate import griddata

# Carregar a imagem
image = cv2.imread('Photos/paisagem.jpego', cv2.IMREAD_GRAYSCALE)

# Exibir a imagem original
cv2.imshow('Imagem Original em Escala de Cinza', image)
cv2.waitKey(0)

# Etapa 1: Suavização da Imagem (Gaussian Blur)
# Usando GaussianBlur para melhor preservação de bordas durante a suavização
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Imagem Suavizada (Gaussian Blur)', smoothed_image)
cv2.waitKey(0)

# Etapa 2: Cálculo da Magnitude do Gradiente com Sobel (ksize=3 para eficiência)
# Usando kernel menor (3x3) para melhor desempenho
sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalizar o gradiente para visualização
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
gradient_magnitude = gradient_magnitude.astype(np.uint8)
cv2.imshow('Magnitude do Gradiente', gradient_magnitude)
cv2.waitKey(0)

# Etapa 3: Thresholding e Supressão de Não-Máximos
# Usando Otsu para thresholding automático
_, thresholded_gradient = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Direção do gradiente em radianos
direction = np.arctan2(sobel_y, sobel_x)
direction = np.degrees(direction)
direction[direction < 0] += 180

# Implementação manual da supressão de não-máximos (melhor controle)
non_max_suppression = np.zeros_like(gradient_magnitude)

for i in range(1, gradient_magnitude.shape[0] - 1):
    for j in range(1, gradient_magnitude.shape[1] - 1):
        angle = direction[i, j]
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            before_pixel = gradient_magnitude[i, j - 1]
            after_pixel = gradient_magnitude[i, j + 1]
        elif (22.5 <= angle < 67.5):
            before_pixel = gradient_magnitude[i - 1, j + 1]
            after_pixel = gradient_magnitude[i + 1, j - 1]
        elif (67.5 <= angle < 112.5):
            before_pixel = gradient_magnitude[i - 1, j]
            after_pixel = gradient_magnitude[i + 1, j]
        else:
            before_pixel = gradient_magnitude[i - 1, j - 1]
            after_pixel = gradient_magnitude[i + 1, j + 1]

        # Aplicar supressão de não-máximos
        if (gradient_magnitude[i, j] >= before_pixel) and (gradient_magnitude[i, j] >= after_pixel):
            non_max_suppression[i, j] = gradient_magnitude[i, j]
        else:
            non_max_suppression[i, j] = 0

cv2.imshow('Bordas Afiladas (Non-Maximum Suppression)', non_max_suppression)
cv2.waitKey(0)

# Etapa 4: Amostragem e Interpolação com Tratamento de Bordas
# Pegando os pontos de borda com os níveis de cinza
sampled_points = []
sampled_values = []

for i in range(0, smoothed_image.shape[0]):
    for j in range(0, smoothed_image.shape[1]):
        if non_max_suppression[i, j] != 0:
            sampled_points.append([i, j])
            sampled_values.append(smoothed_image[i, j])

# Gerar a grade para interpolação
grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]

# Interpolação cúbica com griddata, tratamento de valores de borda
threshold_surface = griddata(sampled_points, sampled_values, (grid_x, grid_y), method='cubic', fill_value=0)

# Normalizar a superfície para exibição
threshold_surface = np.clip(threshold_surface, 0, 255).astype(np.uint8)
cv2.imshow('Superfície de Limiar', threshold_surface)
cv2.waitKey(0)

# Etapa 5: Segmentação
# Comparação pixel a pixel entre a imagem suavizada e a superfície de limiar
segmented_image = np.where(smoothed_image > threshold_surface, 255, 0).astype(np.uint8)

# Exibir a imagem segmentada
cv2.imshow('Imagem Segmentada', segmented_image)
cv2.waitKey(0)

# Fechar todas as janelas
cv2.destroyAllWindows()
