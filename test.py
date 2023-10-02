import cv2
import mediapipe as mp
from time import time

# Webcam width and height
w_camp, h_cam = 640, 480

# Webcam
cap = cv2.VideoCapture(0)

# Set webcam width and height
cap.set(3, w_camp)
cap.set(4, h_cam)

# Previous time and Current time
p_time = 0
c_time = 0

# Hand Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    # Get image from webcam
    success, img = cap.read()
    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process image
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # Get x and y coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 0:
                    # Draw circle on the tip of the index finger
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED) 
                    
            # Draw hand landmarks               
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS Counter
    c_time = time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


    cv2.imshow("Hand Track", img)
    cv2.waitKey(1)

