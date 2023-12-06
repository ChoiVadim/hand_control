import cv2
import math
from time import time
import numpy as np

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandTracking as htm


def main():
    # Volume percentage and volume bar
    volPer = 0
    volBar = 0

    # Volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    # Get volume range
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]

    # Previous time and Current time
    p_time = 0
    
    # Webcam width and height
    w_camp, h_cam = 1280, 720
    
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, w_camp)
    cap.set(4, h_cam)

    # Hand detector
    detecter = htm.handDetector(detectionCon=0.75, trackCon=0.75)

    color = (255, 141, 173)
    while True:
        # Get image from webcam
        success, img = cap.read()
        
        # Find hands
        img = detecter.draw_hands(img)

        # Find position of all landmarks
        lmList = detecter.findPosition(img)

        if len(lmList) != 0:
            # Get x and y coordinates
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw circle on the tip of the index finger
            cv2.circle(img, (x1, y1), 15, (color), 4)
            cv2.circle(img, (x2, y2), 15, (color), 4)
            cv2.circle(img, (cx, cy), 5, (color), cv2.FILLED)

            # Draw line between index finger and thumb
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # Calculate length
            length = math.hypot(x2 - x1, y2 - y1)

            # Interpolate volume
            vol = np.interp(length, [22, 210], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Interpolate volume bar
            volBar = np.interp(length, [22, 210], [30, 517])
            volPer = np.interp(length, [22, 210], [0, 100])

            # Draw circle when length is small
            if length < 50:
                cv2.circle(img, (cx, cy), 25, (color), cv2.FILLED)

        # Draw volume bar
        cv2.rectangle(img, (30, 440), (520, 460), (255, 255, 255), 3)
        cv2.rectangle(img, (33, 443), (int(volBar), 457), (color), cv2.FILLED)
        cv2.putText(img, f"{int(volPer)}%", (540, 460), cv2.FONT_HERSHEY_PLAIN, 2, (color), 2)

        # FPS Counter
        c_time = time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (500, 35), cv2.FONT_HERSHEY_PLAIN, 2, (color), 3)   

        # Show image
        cv2.imshow("Hand Track", img)
        # Exit when press 'q'
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break  

if __name__ == "__main__":
    main()