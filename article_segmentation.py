import cv2
import numpy as np
from scipy.interpolate import griddata

# Carregar a imagem
image = cv2.imread('Photos/paisagem.jpeg', cv2.IMREAD_GRAYSCALE)

# Mostrar a imagem original em escala de cinza
cv2.imshow('Imagem Original em Escala de Cinza', image)
cv2.waitKey(0)

# Etapa 1: Suavização da Imagem (Gaussian Blur)
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Imagem Suavizada (Gaussian Blur)', smoothed_image)
cv2.waitKey(0)

# Etapa 2: Cálculo da Magnitude do Gradiente
gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
gradient_magnitude = gradient_magnitude.astype(np.uint8)
cv2.imshow('Magnitude do Gradiente', gradient_magnitude)
cv2.waitKey(0)

# Etapa 3: Thresholding e Thinning (Supressão de Não-Máximos)
_, thresholded_gradient = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
# Usando Canny como alternativa para thinning
thinned_edges = cv2.Canny(np.uint8(thresholded_gradient), 50, 150)
cv2.imshow('Bordas Afiladas (Thinned Edges)', thinned_edges)
cv2.waitKey(0)

# Etapa 4: Amostragem e Interpolação
edge_points = np.column_stack(np.where(thinned_edges > 0))
gray_levels = smoothed_image[edge_points[:, 0], edge_points[:, 1]]
grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
threshold_surface = griddata(edge_points, gray_levels, (grid_x, grid_y), method='cubic', fill_value=0)
threshold_surface = np.clip(threshold_surface, 0, 255).astype(np.uint8)
cv2.imshow('Superfície de Limiar', threshold_surface)
cv2.waitKey(0)

# Etapa 5: Segmentação
segmented_image = np.where(smoothed_image > threshold_surface, 255, 0).astype(np.uint8)
cv2.imshow('Imagem Segmentada', segmented_image)
cv2.waitKey(0)

# Fechar todas as janelas abertas
cv2.destroyAllWindows()
