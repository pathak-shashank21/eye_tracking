import cv2
import numpy as np
import pyautogui

def gaze_dot_overlay(pipe):
    screen_w, screen_h = pyautogui.size()
    window_name = "Gaze Overlay"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    dot_radius = 20

    while True:
        overlay = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        if pipe.poll():
            data = pipe.recv()
            if data == "_END_":
                break
            x, y = map(int, data)
            cv2.circle(overlay, (x, y), dot_radius, (100, 200, 255), -1)
        cv2.imshow(window_name, overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
