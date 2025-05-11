import socket
import json
import time
import csv
import cv2
import numpy as np
from input_reader import get_screen_size

# Initialize screen size
screen_w, screen_h = get_screen_size()

# Set up UDP socket to receive data from OpenSeeFace
UDP_IP = "127.0.0.1"
UDP_PORT = 11574
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2)

# Calibration points (simple grid on screen)
calib_points = [
    (screen_w // 6, screen_h // 6), (screen_w // 2, screen_h // 6), (5 * screen_w // 6, screen_h // 6),
    (screen_w // 6, screen_h // 2), (screen_w // 2, screen_h // 2), (5 * screen_w // 6, screen_h // 2),
    (screen_w // 6, 5 * screen_h // 6), (screen_w // 2, 5 * screen_h // 6), (5 * screen_w // 6, 5 * screen_h // 6)
]

cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Open CSV file for calibration data
with open("calibration_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["landmark_1_x", "landmark_1_y", "landmark_1_z", "landmark_2_x", "landmark_2_y", "landmark_2_z", "calib_x", "calib_y"])

    for cx, cy in calib_points:
        print(f"🎯 Look at point ({cx}, {cy}) and blink to confirm.")
        stable_start = time.time()
        blink_streak = 0  
        point_received = False  

        while not point_received:
            screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(screen, (cx, cy), 30, (0, 255, 255), -1)

            try:
                data, _ = sock.recvfrom(65536)
                face_data = json.loads(data.decode("utf-8"))

                print(f"📡 Faces received: {len(face_data)}")

                if isinstance(face_data, list) and len(face_data) > 0:
                    face = face_data[0]

                    if time.time() - stable_start < 1:
                        cv2.imshow("Calibration", screen)
                        if cv2.waitKey(1) == 27:
                            print("❌ Calibration aborted.")
                            break
                        continue

                    is_blinking = face.get("is_blinking", False)
                    if is_blinking:
                        blink_streak += 1
                    else:
                        blink_streak = 0

                    print(f"👁️ Blinking: {is_blinking} | Blink Streak: {blink_streak}")

                    if blink_streak >= 3:
                        print("✅ Blink confirmed.")
                        features = face.get('landmarks3d', [])
                        if len(features) >= 6:
                            row = list(features[:6]) + [cx, cy]  
                            writer.writerow(row)

                            cv2.circle(screen, (cx, cy), 40, (0, 255, 0), -1)
                            cv2.imshow("Calibration", screen)
                            cv2.waitKey(500)  
                            point_received = True 

                        else:
                            print("❌ Insufficient landmarks received.")
                            continue

                else:
                    print("❌ No valid face data received.")
                    continue

            except socket.timeout:
                print("❌ No data received within timeout period.")
                continue

            except json.JSONDecodeError as e:
                print(f"❌ JSON decoding error: {e}") 
                continue

            except Exception as e:
                print(f"❌ Unexpected error: {e}")  
                continue

            cv2.imshow("Calibration", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("❌ Calibration aborted by user.")
                break

cv2.destroyAllWindows()
print("✅ Calibration complete.")
