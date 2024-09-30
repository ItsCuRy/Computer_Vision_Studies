import cv2
import numpy as np

# Pegando a imagem da webcam
video = cv2.VideoCapture(0)

# Pre-processamento da imagem
def preProcess(img):
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR para preto e branco , opencv interpreta a imagem em bgr 
    imgBlur = cv2.GaussianBlur(imgGray, (15, 15), 0) #filtro gaussiano de desfoque para reduzir ruidos 
    _, imgThresh = cv2.threshold(imgBlur, 130, 255, cv2.THRESH_BINARY_INV) #threshold binario inverso, pois geralmente as moedas sao mais escuras que o fundo
    return imgThresh

# Separar objetos conectados com a Transformada de Distância
def separate_connected_objects(imgPre):
    # Aplicar a Transformada de Distância
    dist_transform = cv2.distanceTransform(imgPre, cv2.DIST_L2, 5) #calcula a distancia de cada pixel branco ate o pixel preto mais proximo, distancia euclidiana onde os centros tem valores mais altos 
    # Normalizar e aplicar um limiar para detectar os picos (centros das moedas)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Aumentar as áreas de fundo
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(imgPre, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcar as moedas com base nos centros detectados
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Aplicar Watershed para separar objetos conectados
    img3 = cv2.cvtColor(imgPre, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img3, markers)
    imgPre[markers == -1] = 0  # Marcar as bordas

    return imgPre

# Função para filtrar contornos baseados em área e circularidade
def is_coin(contour):
    area = cv2.contourArea(contour)
    if area < 500 or area > 3000:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    if circularity > 0.7:
        return True
    return False

while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))
    
    # Pre-processa a imagem
    imgPre = preProcess(img)

    # Separar moedas conectadas
    imgPre = separate_connected_objects(imgPre)

    # Detecta contornos
    contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Percorre os contornos e filtra os que são moedas
    for cnt in contours:
        if is_coin(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Moeda", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostra a imagem original com os contornos detectados
    cv2.imshow('Imagem', img)
    # Mostra a imagem pre-processada
    cv2.imshow('PreProcessada', imgPre)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
video.release()
cv2.destroyAllWindows()
