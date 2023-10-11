from time import time, sleep
import numpy as np

import cv2
import mediapipe as mp
from cvzone.FPS import FPS


TEXT_COLOR = (0, 0, 0)
DOT_COLOR = (0, 125, 0)
LINE_COLOR = (255, 255, 255)

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.modelC = modelC
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Hand Detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def draw_hands(self, img , draw=True):

        def get_label(index, hand, results):
            output = None
            for classification in results.multi_handedness:
                if classification.classification[0].index == index:
                    label = classification.classification[0].label
                    score = classification.classification[0].score
                    text = '{} {}'.format(label, round(score, 2))

                    coords = tuple(np.multiply(
                        np.array((hand.landmark[0].x, hand.landmark[0].y)),
                    [640, 480]).astype(int))

                    output = text, coords
            return output
                
        # Convert image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image
        self.results = self.hands.process(imgRGB)

        # If hands are detected
        if self.results.multi_hand_landmarks:
                        
            for num, handLms in enumerate(self.results.multi_hand_landmarks):
                if len(self.results.multi_handedness) == 1:
                    label = self.results.multi_handedness[0].classification[0].label    
                    if label == "Left":
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=DOT_COLOR, thickness=4, circle_radius=3),
                        self.mpDraw.DrawingSpec(color=LINE_COLOR, thickness=1, circle_radius=0))
                    if label == "Right":
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=DOT_COLOR, thickness=4, circle_radius=7),
                        self.mpDraw.DrawingSpec(color=LINE_COLOR, thickness=3, circle_radius=0))
                if len(self.results.multi_handedness) == 2:
                    label = self.results.multi_handedness[num].classification[0].label
                    if num == 0 and label == "Left":
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=DOT_COLOR, thickness=4, circle_radius=3),
                        self.mpDraw.DrawingSpec(color=LINE_COLOR, thickness=1, circle_radius=0))
                    if num == 1 and label == "Right":
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=DOT_COLOR, thickness=4, circle_radius=7),
                        self.mpDraw.DrawingSpec(color=LINE_COLOR, thickness=3, circle_radius=0))

                if get_label(num, handLms, self.results):
                    text, coords = get_label(num, handLms, self.results)
                    cv2.putText(img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        return img
    

    def findPosition(self, img, handNo=0):
        lmList = []
        # Check if there is hand
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # Get x and y coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList
    

    def find_pos_hands(self, img):
        left_hand_lmList = []
        right_hand_lmList = []
        lmList = []
        # Check if there is hands
        if self.results.multi_hand_landmarks:

                for num, hand in enumerate(self.results.multi_hand_landmarks):

                    if len(self.results.multi_handedness) == 1:
                        for id, lm in enumerate(hand.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([id, cx, cy])
                        return [lmList]

                    if len(self.results.multi_handedness) == 2:
                        label = self.results.multi_handedness[num].classification[0].label

                        if num == 0 and label == "Left":
                            for id, lm in enumerate(hand.landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                left_hand_lmList.append([id, cx, cy])

                        if num == 1 and label == "Right":
                            for id, lm in enumerate(hand.landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                right_hand_lmList.append([id, cx, cy])

        return [left_hand_lmList, right_hand_lmList]
                
    
def main():

    # # Webcam width and height
    # w_camp, h_cam = 640, 480
    # # Set webcam width and height
    # cap.set(3, w_camp)
    # cap.set(4, h_cam)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Set the frames per second to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize hand detector
    hand_detecter = handDetector()

    fpsReader = FPS(avgCount=30)

    while True:
        # Get image from webcam
        success, img = cap.read()
        # Flip image
        img = cv2.flip(img, 1)

        fps, img = fpsReader.update(img, pos=(20, 50), bgColor=(255, 0, 255),
                                    textColor=(255, 255, 255), scale=3, thickness=3)          
        
        # Find hands
        img = hand_detecter.draw_hands(img)


        #Find position
        lmList = hand_detecter.find_pos_hands(img)

        if lmList:
            if len(lmList) == 1:
                print(f"Hand: {lmList[0][4]}")
            if len(lmList) == 2:
                if lmList[0] and lmList[1] != []:
                    print(f"Right Hand: {lmList[1][4]}")
                    print(f"Left Hand: {lmList[0][4]}")


        # Show image
        cv2.imshow("Hand Track", img)
        # Exit when press "esc"
        if cv2.waitKey(1) & 0xff == ord('q'):
            break    

        
if __name__ == "__main__":
    main()