import socket
import json
import numpy as np

class FaceTracker:
    def __init__(self, model_dir, port=11573):  # Ensure it's listening on port 11573
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))  # Listening on 11573
        self.sock.settimeout(0.1)  # Set socket timeout to avoid blocking indefinitely
        self.model_dir = model_dir

    def detect_faces(self, frame):
        """
        Receives and processes face data from the OpenSeeFace UDP stream.
        """
        try:
            data, _ = self.sock.recvfrom(65536)  # Receive UDP data
            face_data = json.loads(data.decode("utf-8"))  # Decode the incoming data into JSON format

            # Debugging: print the raw data
            print("📡 Raw UDP Data Received:", face_data)

            # Check if the face data is a list (multiple faces) or dictionary (single face)
            if isinstance(face_data, list):
                return face_data  # Return list of faces
            elif isinstance(face_data, dict):
                # Return single face data wrapped in a list
                return [face_data]
            else:
                print("❌ Invalid data format received.")
                return []  # Return an empty list if the data is neither a list nor a dictionary

        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding error: {e}")  # Handle JSON decoding errors
            return []
        except Exception as e:
            print(f"❌ Error receiving data: {e}")  # General exception handling
            return []

    def get_3d_features(self, face):
        """
        Extracts 3D landmarks and returns key facial features as [x1, y1, x2, y2, ...] for gaze prediction.
        """
        landmarks_3d = face.get('landmarks3d', [])
        
        # Ensure that the landmarks are valid
        def safe_get(idx):
            if idx < len(landmarks_3d):
                return landmarks_3d[idx]
            return [0.0, 0.0, 0.0]  # Default to zeros if landmark is missing
        
        # Extract relevant features
        left_eye = safe_get(33)  # Left eye landmark
        right_eye = safe_get(133)  # Right eye landmark
        nose_tip = safe_get(1)  # Nose tip landmark
        
        # Return the relevant features for gaze prediction
        return [
            left_eye[0], left_eye[1],  # Left eye (x, y)
            right_eye[0], right_eye[1],  # Right eye (x, y)
            nose_tip[0], nose_tip[1]  # Nose tip (x, y)
        ]

    def get_blink_status(self, face):
        """
        Uses Eye Aspect Ratio (EAR) to detect if the person is blinking.
        A simple threshold is used to detect the blink.
        """
        landmarks = face.get('landmarks2d', [])
        
        if len(landmarks) < 6:
            return False  # Not enough landmarks for blink detection

        # Eye Aspect Ratio (EAR) calculation
        left_eye = landmarks[36:42]  # Left eye landmarks (6 points)
        right_eye = landmarks[42:48]  # Right eye landmarks (6 points)

        # Calculate EAR for left eye
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # Average EAR value
        ear = (left_ear + right_ear) / 2.0
        print(f"EAR: {ear}")

        # Blink detection threshold
        if ear < 0.2:
            return True  # Detected blink (EAR below threshold)
        return False  # No blink detected

    def calculate_ear(self, eye_points):
        """
        Calculates the Eye Aspect Ratio (EAR) for a set of eye landmarks.
        EAR is used to determine if a person is blinking.
        """
        # Calculate distances between vertical and horizontal eye landmarks
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear

    def track_faces_and_blinks(self, frame):
        """
        Tracks faces, extracts gaze-related 3D features, and detects blinks.
        This is the main function to be called in a loop for real-time tracking.
        """
        faces = self.detect_faces(frame)
        
        all_features = []
        for face in faces:
            # Extract 3D features for gaze prediction
            features = self.get_3d_features(face)
            all_features.append(features)

            # Detect blink status
            is_blinking = self.get_blink_status(face)
            print(f"Blink Detected: {is_blinking}")
        
        return all_features

# Example usage:
if __name__ == "__main__":
    tracker = FaceTracker(model_dir='/path/to/your/model/dir')
    while True:
        # Here `frame` should be the current frame from your webcam or video stream
        frame = None  # Replace with your actual frame source
        gaze_features = tracker.track_faces_and_blinks(frame)
        print(gaze_features)
