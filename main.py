import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import pyautogui

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Inicializar o detector de mãos da biblioteca cvzone
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Limite de distância para detectar toque
TOUCH_THRESHOLD = 50  # Limite para detectar o toque

# Variável para armazenar a posição y anterior
previous_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverter o frame horizontalmente
    frame = cv2.flip(frame, 1)

    # Detectar mãos na imagem
    hands, frame = detector.findHands(frame)

    if hands:
        for hand in hands:
            hand_points = hand["lmList"]  # Lista de pontos chave
            thumb_tip = hand_points[4]
            index_tip = hand_points[8]
            middle_tip = hand_points[12]
            ring_tip = hand_points[16]  # Anelar

            # Desenhar os pontos-chave
            for i, point in enumerate(hand_points):
                color = (255, 0, 0) if i not in [4, 8, 12, 16] else (0, 255, 0)
                cv2.circle(frame, (point[0], point[1]), 5, color, -1)

            # Verificar se os dedos indicador e polegar estão tocando
            distance_index_thumb = euclidean_distance(index_tip, thumb_tip)
            if distance_index_thumb < TOUCH_THRESHOLD:
                # Desenhar as coordenadas no canto superior esquerdo
                coordinates_text = f"X: {index_tip[0]} Y: {index_tip[1]}"
                cv2.putText(frame, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Realizar a rolagem da tela baseada na posição y
                if previous_y is not None:
                    scroll_amount = (previous_y - index_tip[1]) / 4  # Ajuste o divisor para controlar a sensibilidade
                    pyautogui.scroll(scroll_amount)
                
                previous_y = index_tip[1]
            else:
                previous_y = None  # Resetar se os dedos não estiverem tocando

            # Verificar se os dedos médio e polegar estão tocando
            distance_middle_thumb = euclidean_distance(middle_tip, thumb_tip)
            if distance_middle_thumb < TOUCH_THRESHOLD:
                # Calcular a posição do mouse entre o polegar e o dedo médio
                mouse_x = (thumb_tip[0] + middle_tip[0]) // 2
                mouse_y = (thumb_tip[1] + middle_tip[1]) // 2
                
                # Mapeia as coordenadas para a tela
                screen_width, screen_height = pyautogui.size()
                mouse_x = np.interp(mouse_x, [0, frame.shape[1]], [0, screen_width])
                mouse_y = np.interp(mouse_y, [0, frame.shape[0]], [0, screen_height])
                pyautogui.moveTo(mouse_x, mouse_y)

            # Verificar se o polegar e o anelar estão tocando para clicar
            distance_ring_thumb = euclidean_distance(ring_tip, thumb_tip)
            if distance_ring_thumb < TOUCH_THRESHOLD:
                # Simular um clique do mouse
                pyautogui.click()

    # Exibir o frame resultante
    cv2.imshow('Hand Tracking', frame)

    # Sair do loop se a tecla 'esc' for pressionada
    key = cv2.waitKey(1)
    if key == 27:
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
