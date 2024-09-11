import cv2
import numpy as np

# Pegando a imagem da webcam
video = cv2.VideoCapture(0)

# Pre-processamento da imagem
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (15, 15), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    kernel = np.ones((5, 5))
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=1)
    return imgDilated

# Função para filtrar contornos baseados em área e circularidade
def is_coin(contour):
    area = cv2.contourArea(contour)
    if area < 500 or area > 3000:  # Ajuste estes valores conforme necessário
        return False
    
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    if circularity > 0.7:  # Um valor próximo de 1 indica uma forma mais circular
        return True
    return False

while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))
    
    # Pre-processa a imagem
    imgPre = preProcess(img)

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
