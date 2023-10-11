import threading

import cv2
from deepface import DeepFace

import numpy as np

img = cv2.VideoCapture(0, cv2.CAP_DSHOW)

img.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

face_match = False

reference_img = cv2.imread("reference.jpg")


def check_face(frame):
    global face_match

    try:
        result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
        print(result)
        if result["verified"]:
            face_match = True
            print("Matched")
        else:
            face_match = False
            print("Not Matched")

    except ValueError:
        print("No face found")
        face_match = False

    

while True:

    return_value, frame = img.read()
    frame = cv2.flip(frame, 1)

    if return_value:
        if count % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(), )).start()
            except ValueError:
                pass
        count += 1
    
    if face_match:
        cv2.putText(frame, "Matched", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, "Hello", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not Matched", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("Jarvis", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()