import cv2
import socket

import HandTracking as htm

# The MAC address of a Bluetooth adapter on the server
HOST = 'C0:3C:59:D8:CE:8E'
# The port used by the server
PORT = 5

def main():

    # Webcam width and height
    w_camp, h_cam = 640, 480
    
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, w_camp)
    cap.set(4, h_cam)

    # Hand detector
    detecter = htm.handDetector()


    # Create the server socket
    server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server.bind((HOST, PORT))
    server.listen(2)

    # Accept connections from the second client
    wheels_socket, address2 = server.accept()
    print('Connected by ', address2)
    message = wheels_socket.recv(1024).decode("utf-8")
    print('Received message: ', message)

    list = [0, 0, 0, 0, 0]

    while True:
        # Get image from webcam
        success, img = cap.read()
        
        # Find hands
        img = detecter.draw_hands(img)

        # Find position
        lmList = detecter.findPosition(img)
        if len(lmList) != 0:
            if lmList[8][2] < lmList[6][2]:
                list[1] = 1
            if lmList[12][2] < lmList[10][2]:
                list[2] = 1
            if lmList[16][2] < lmList[14][2]:
                list[3] = 1
            if lmList[20][2] < lmList[18][2]:
                list[4] = 1
            if lmList[4][1] > lmList[3][1]:
                list[0] = 1

            if lmList[8][2] > lmList[6][2]:
                list[1] = 0
            if lmList[12][2] > lmList[10][2]:
                list[2] = 0
            if lmList[16][2] > lmList[14][2]:
                list[3] = 0
            if lmList[20][2] > lmList[18][2]:
                list[4] = 0
            if lmList[4][1] < lmList[3][1]:
                list[0] = 0

        try:
            message = "".join(str(x) for x in list)
            print(message)
            wheels_socket.send(f"{message}".encode("ASCII"))
        except ConnectionAbortedError:
            print("Connection aborted")
            break

        cv2.imshow("Hand Track", img)
        cv2.waitKey(1)    

if __name__ == "__main__":
    main()