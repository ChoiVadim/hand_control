import cv2
import face_recognition

face_match = False


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    known_image = face_recognition.load_image_file("reference.jpg")
    unknown_image = face_recognition.load_image_file(img)

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

    print(results)

    if face_match:
        cv2.putText(img, "Matched", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(img, "Hello", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Not Matched", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("Jarvis", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()