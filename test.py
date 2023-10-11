import socket
import threading
import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector


# The MAC address of a Bluetooth adapter on the server
HOST = 'C0:3C:59:D8:CE:8E'
# The port used by the server
PORT = 6
# The size of the header
HEADER_SIZE = 64
# The format of the message
FORMAT = "ASCII"


def main():

    print("Starting server...")
    # Create the server socket
    server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Server is listening on {HOST}")

    # Accept connections from the first client
    robot_arm_socket, address = server.accept()
    print('Connected by ', address)
    message = robot_arm_socket.recv(HEADER_SIZE).decode(FORMAT)
    print('Received message: ', message)

    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Flip the frame horizontally for easier use

        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector.findHands(img, draw=True, flipType=False)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            # print(f'H1 = {fingers1}', end=" ")  # Print the count of fingers that are up
            if fingers1:
                msg_to_arm = "".join(str(i) for i in fingers1)

                input_x = np.array([0, 640])
                output_x = np.array([-300, 300])

                input_y = np.array([0, 480])
                output_y = np.array([0, -300])

                # Find the value of gold on day 3
                x_value = np.interp(center1[0], input_x, output_x)
                y_value = np.interp(center1[1], input_y, output_y)

                
                msg_to_arm += f" {int(x_value)} {int(y_value)}"

                print(msg_to_arm)
                message = msg_to_arm.encode(FORMAT)
                msg_length = len(message)
                send_length = str(msg_length).encode(FORMAT)
                send_length += b' ' * (HEADER_SIZE - len(send_length))

                robot_arm_socket.send(send_length)
                robot_arm_socket.send(message)
                # Center of hand
                cv2.circle(img, center1, 10, (0, 0, 255), cv2.FILLED)

                # Send the message to the client


            # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                      scale=10)

            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingersUp(hand2)

                # Calculate distance between the index fingers of both hands and draw it on the image
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                          scale=10)
                
                


        # Display the image in a window
        cv2.imshow("Hand Track", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        if cv2.waitKey(1) & 0xff == ord('q'):
            robot_arm_socket.close()
            break


if __name__ == "__main__":
    main()