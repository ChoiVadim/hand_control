import cv2
import socket
import threading

import HandTracking as htm


# The MAC address of a Bluetooth adapter on the server
HOST = 'C0:3C:59:D8:CE:8E'
# The port used by the server
PORT = 5
# The size of the header
HEADERSIZE = 64
# The format of the message
FORMAT = "ASCII"
# The message to be sent
MESSAGE = [0, 0, 0, 0, 0]
# Webcam width and height
w_cam, h_cam = 640, 480


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    msg_from_client = conn.recv(HEADERSIZE).decode(FORMAT)
    print(f"[{addr}] {msg_from_client}")

    while connected:
        try:
            msg = "".join(str(x) for x in MESSAGE)
            conn.send(f"{msg}".encode("FORMAT"))

        except ConnectionAbortedError:
            print("Connection aborted")
            connected = False
            conn.close()


def start_server():
    print("Starting server...")
    # Create the server socket
    server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server.bind((HOST, PORT))
    server.listen(2)
    print(f"Server is listening on {HOST}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print("Active connections: ", threading.activeCount() - 1)


def start_detector():
    global MESSAGE
    
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    # Hand detector
    detecter = htm.handDetector()

    while True:
        # Get image from webcam
        success, img = cap.read()

        # Flip image
        img = cv2.flip(img, 1)
        
        # Find hands
        img = detecter.draw_hands(img)

        # Find position
        lmList = detecter.findPosition(img)
        if len(lmList) != 0:
            if lmList[8][2] < lmList[6][2]:
                MESSAGE[1] = 1
            if lmList[12][2] < lmList[10][2]:
                MESSAGE[2] = 1
            if lmList[16][2] < lmList[14][2]:
                MESSAGE[3] = 1
            if lmList[20][2] < lmList[18][2]:
                MESSAGE[4] = 1
            if lmList[4][1] > lmList[3][1]:
                MESSAGE[0] = 1

            if lmList[8][2] > lmList[6][2]:
                MESSAGE[1] = 0
            if lmList[12][2] > lmList[10][2]:
                MESSAGE[2] = 0
            if lmList[16][2] > lmList[14][2]:
                MESSAGE[3] = 0
            if lmList[20][2] > lmList[18][2]:
                MESSAGE[4] = 0
            if lmList[4][1] < lmList[3][1]:
                MESSAGE[0] = 0

        cv2.imshow("Hand Track", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break   


def main():
    # threading.Thread(target=start_server).start()
    start_detector()

if __name__ == "__main__":
    main()