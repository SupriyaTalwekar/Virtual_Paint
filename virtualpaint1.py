import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
draw_color = colors[0]
xp, yp = 0, 0
color_index = 0
cooldown = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def fingers_up(hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        fingers.append(hand.landmark[tip_ids[i]].y < hand.landmark[tip_ids[i]-2].y)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm_list = handLms.landmark
            fingers = fingers_up(handLms)
            x1 = int(lm_list[8].x * 1280)
            y1 = int(lm_list[8].y * 720)

            if fingers[1] and not any(fingers[2:]):
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(img, (xp, yp), (x1, y1), draw_color, 10)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 10)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

            if fingers[1] and fingers[2] and not fingers[3] and cooldown == 0:
                color_index = (color_index + 1) % len(colors)
                draw_color = colors[color_index]
                cooldown = 20

            if fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "Cleared", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    if cooldown > 0:
        cooldown -= 1

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Paint", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
