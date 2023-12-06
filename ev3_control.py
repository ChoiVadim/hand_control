import socket

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FPS import FPS


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
    server = socket.socket(socket.AF_BLUETOOTH,
                            socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server.bind((HOST, PORT))
    server.listen(2)
    print(f"Server is listening on {HOST}")

    # Accept connections from the first client
    robot_arm_socket, arm_address = server.accept()
    print('Connected by ', arm_address)
    message = robot_arm_socket.recv(HEADER_SIZE).decode(FORMAT)
    print('Received message: ', message)

    # Accept connections from the second client
    wheels_socket, wheels_address = server.accept()
    print('Connected by ', wheels_address)
    message = wheels_socket.recv(HEADER_SIZE).decode(FORMAT)
    print('Received message: ', message)


    # Initialize the webcam to capture video
    cap = cv2.VideoCapture(0)
    w_camp, h_cam = 1280, 720
    cap.set(3, w_camp)
    cap.set(4, h_cam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fpsReader = FPS(avgCount=30)

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

        # Update the FPS counter and display it on the image
        fps, img = fpsReader.update(img, pos=(20, 50),
                            bgColor=(255, 0, 255), textColor=(255, 255, 255),
                            scale=3, thickness=3)

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
            # Coordinates of the dot number 20 on the hand
            dot = lmList1[0][0:2]

            if fingers1:
                # Message to be sent to the client (fingers up or down)
                msg_to_arm = "".join(str(i) for i in fingers1)

                # Convert the coordinates of the dot on the hand to coordinates on the screen
                input_x = np.array([300, 1240])
                output_x = np.array([-350, 350])
                input_y = np.array([100, 680])
                output_y = np.array([-400, 0])
                # Interpolate the x and y values
                x_value = np.interp(dot[0], input_x, output_x)
                y_value = np.interp(dot[1], input_y, output_y)

                # Add the coordinates of dot to the message
                msg_to_arm += f" {int(x_value)} {int(y_value) }"

                # Convert the message to bytes and send it to the client
                # You need to send a length header before sending the message itself
                # Because if you don't send a header, the client won't know how long the message is
                message = msg_to_arm.encode(FORMAT)
                msg_length = len(message)
                send_length = str(msg_length).encode(FORMAT)
                send_length += b' ' * (HEADER_SIZE - len(send_length))

                # Send the message length first and then the message
                robot_arm_socket.send(send_length)
                robot_arm_socket.send(message)

                # Information for the second hand detected
                recv_msg = robot_arm_socket.recv(HEADER_SIZE).decode(FORMAT)
                distance = f"Distance: {recv_msg}cm"

                # Display the distance on the screen
                img = cv2.putText(img, distance, (20, 100), cv2.FONT_HERSHEY_SIMPLEX ,  
                                    1, (255, 255, 255) , 2, cv2.LINE_AA) 

                # Draw a rectangle hand detection area
                img = cv2.rectangle(img, (300, 100), (1240, 680), (255, 0, 255), 2)
                # Dot on the hand
                cv2.circle(img, dot, 10, (0, 0, 255), cv2.FILLED)

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

                if fingers2:
                    # Message to be sent to the client (fingers up or down)
                    msg_to_wheels = "".join(str(i) for i in fingers2)

                    message = msg_to_wheels.encode(FORMAT)
                    msg_length = len(message)
                    send_length = str(msg_length).encode(FORMAT)
                    send_length += b' ' * (HEADER_SIZE - len(send_length))

                    # Send the message length first and then the message
                    wheels_socket.send(send_length)
                    wheels_socket.send(message)
                    
        if not hands:
            # Send a stop msg to the wheels
            stop_msg = "11111"
            msg_length = len(stop_msg)
            send_length = str(msg_length).encode(FORMAT)
            send_length += b' ' * (HEADER_SIZE - len(send_length))
            wheels_socket.send(send_length)
            wheels_socket.send(stop_msg.encode(FORMAT))

            # Send a stop msg to the arms
            stop_msg = "00000 0 0"
            msg_length = len(stop_msg)
            send_length = str(msg_length).encode(FORMAT)
            send_length += b' ' * (HEADER_SIZE - len(send_length))
            robot_arm_socket.send(send_length)
            robot_arm_socket.send(stop_msg.encode(FORMAT))

        # Display the image in a window
        cv2.imshow("Hand Track", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        if cv2.waitKey(1) & 0xff == ord('q'):
            robot_arm_socket.close()
            break

if __name__ == "__main__":
    main()
