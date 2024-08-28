import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_adaptive_threshold(img):
    # Aplicar threshold adaptativo
    threshold_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return threshold_img

def split_image_and_show_separated(image_path, padding=10):
    # Carregar a imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise Exception(f"Erro ao carregar a imagem: {image_path}. Verifique o caminho e a integridade do arquivo.")
    
    height, width = img.shape
    
    # Número de partes em cada dimensão (4x4 = 16 partes)
    parts_x, parts_y = 4, 4
    part_width = width // parts_x
    part_height = height // parts_y

    # Definir tamanho da nova imagem com espaço (padding) entre as partes
    new_width = parts_x * part_width + (parts_x - 1) * padding
    new_height = parts_y * part_height + (parts_y - 1) * padding
    
    # Criar nova imagem (preenchida com fundo preto)
    new_img = np.zeros((new_height, new_width), dtype=np.uint8)

    # Dividir e aplicar threshold adaptativo a cada parte e colocar na nova imagem com espaçamento
    for i in range(parts_x):
        for j in range(parts_y):
            # Definir as coordenadas da parte atual
            left = i * part_width
            top = j * part_height
            right = left + part_width
            bottom = top + part_height
            
            # Recortar a parte atual da imagem
            part = img[top:bottom, left:right]
            
            # Aplicar o threshold adaptativo à parte
            part_with_threshold = apply_adaptive_threshold(part)
            
            # Coordenadas para a nova imagem
            new_left = i * (part_width + padding)
            new_top = j * (part_height + padding)
            
            # Inserir a parte processada na nova imagem
            new_img[new_top:new_top + part_height, new_left:new_left + part_width] = part_with_threshold

    # Mostrar a nova imagem gerada
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    plt.show()

# Caminho para a imagem
image_path = r'C:\Users\arthu.PC-DO-ARTHUR\OneDrive\Imagens\Saved Pictures\rosto.jpg'
split_image_and_show_separated(image_path)
