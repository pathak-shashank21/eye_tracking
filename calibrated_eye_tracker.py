import cv2
import numpy as np
from facetracker import FaceTracker
from gaze_model import GazeRegressor
from input_reader import get_screen_size
from collections import deque

# Load the trained gaze model
model = GazeRegressor()
model.load("calibration_data.csv")

# Initialize face tracker and screen
face_tracker = FaceTracker("models")
screen_w, screen_h = get_screen_size()

# Create a full-screen window for gaze tracking
cv2.namedWindow("Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Smoothing buffers to stabilize gaze dot movement
smooth_x = deque(maxlen=5)  # Buffer to smooth x-coordinate
smooth_y = deque(maxlen=5)  # Buffer to smooth y-coordinate
trail = deque(maxlen=20)  # Buffer to store trail history for gaze point

# Blink detection (optional)
blink_threshold = 0.2  # EAR threshold for detecting blinks

# Function for blink detection (optional)
def detect_blink(face):
    return face_tracker.get_blink_status(face)

while True:
    screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)  # Black screen for drawing gaze
    faces = face_tracker.detect_faces(None)
    print("📡 Faces received:", len(faces))

    if faces:
        face = faces[0]
        try:
            # Optional: Blink detection
            is_blinking = detect_blink(face)
            if is_blinking:
                print("👀 Blink detected!")

            # Extract 3D landmarks and predict gaze
            features = face_tracker.get_3d_features(face)
            x, y = model.predict(features)

            # Clip the gaze point within screen bounds
            x = int(np.clip(x, 0, screen_w - 1))
            y = int(np.clip(y, 0, screen_h - 1))

            # Smooth the gaze point by averaging over a buffer of previous points
            smooth_x.append(x)
            smooth_y.append(y)
            avg_x = int(np.mean(smooth_x))
            avg_y = int(np.mean(smooth_y))

            # Add smoothed gaze point to the trail buffer for visualization
            trail.append((avg_x, avg_y))

            # Draw the gaze trail (dots with fading effect)
            for i, (tx, ty) in enumerate(trail):
                alpha = i / len(trail)  # Fading effect
                radius = int(10 + 10 * alpha)  # Gradual increase in size
                color = (255, int(100 + 100 * alpha), 100)  # Gradual color shift
                cv2.circle(screen, (tx, ty), radius, color, -1)

        except Exception as e:
            print("⚠️ Gaze prediction failed:", e)

    # Display the gaze tracker window
    cv2.imshow("Gaze Tracker", screen)

    # Exit if the user presses the "Esc" key
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
