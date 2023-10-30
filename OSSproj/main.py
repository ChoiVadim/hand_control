from multiprocessing import Process

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PoseModule import PoseDetector
from cvzone.Utils import stackImages
from cvzone.FPS import FPS


fpsReader = FPS(avgCount=30)

def face_re():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        face_detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
        img, bboxs = face_detector.findFaces(img, draw=True)
        # FPS
        fps, img = fpsReader.update(img, pos=(20, 50),
                            bgColor=(255, 0, 255), textColor=(255, 255, 255),
                            scale=3, thickness=3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def face_mesh():
    while True:
        cap = cv2.VideoCapture(1)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        success, img = cap.read()
        img = cv2.flip(img, 1)

        face_mesh_detector = FaceMeshDetector(maxFaces=1)
        img, faces = face_mesh_detector.findFaceMesh(img, draw=True)
        # FPS
        fps, img = fpsReader.update(img, pos=(20, 50),
                            bgColor=(255, 0, 255), textColor=(255, 255, 255),
                            scale=3, thickness=3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def hand():
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
        hands, img = hand_detector.findHands(img, draw=True)
        # FPS
        fps, img = fpsReader.update(img, pos=(20, 50),
                            bgColor=(255, 0, 255), textColor=(255, 255, 255),
                            scale=3, thickness=3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()

def pose():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        
        success, img = cap.read()
        img = cv2.flip(img, 1)

        pose_detector = PoseDetector()
        img = pose_detector.findPose(img, draw=True)

        # FPS
        fps, img = fpsReader.update(img, pos=(20, 50),
                            bgColor=(255, 0, 255), textColor=(255, 255, 255),
                            scale=3, thickness=3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    cv2.destroyAllWindows()

def main():
    Process(target=face_re).start()
    Process(target=pose).start()

if __name__ == "__main__":
    main()
    
    # imgList = [hand(img.copy()), face_mesh(img.copy())]
    # imgStacked = stackImages(imgList, 2, 0.8)
    # cv2.imshow("stackedImg", imgStacked)
