import tkinter as tk
import socket
import json
import numpy as np

def get_screen_size():
    """Get the screen resolution."""
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    return root.winfo_screenwidth(), root.winfo_screenheight()

# OpenSeeFace UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 11573

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Set non-blocking mode

def process_face_data(face_data):
    """
    Process and extract 3D landmark data from the incoming face data.
    """
    # Check if the face data contains 3D landmarks
    if "landmarks3d" in face_data:
        landmarks = face_data["landmarks3d"]

        # Ensure landmarks are valid, should have at least 12 values (representing the 3D positions)
        if len(landmarks) >= 12:
            return np.array(landmarks[:12])  # Return first 12 landmarks for gaze prediction
    return None

def get_face():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 11573
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    def process_face_data(face_data):
        if "landmarks3d" in face_data:
            landmarks = face_data["landmarks3d"]
            if len(landmarks) >= 12:
                return np.array(landmarks[:12])
        return None

    try:
        data, _ = sock.recvfrom(65536)
        face_data = json.loads(data.decode('utf-8'))
        if isinstance(face_data, dict) and "faces" in face_data:
            faces = face_data["faces"]
            processed_faces = []
            for face in faces:
                landmarks = process_face_data(face)
                if landmarks is not None:
                    processed_faces.append(landmarks)
            return processed_faces
        elif isinstance(face_data, dict) and "landmarks3d" in face_data:
            landmarks = process_face_data(face_data)
            if landmarks is not None:
                return [landmarks]
        return []
    except BlockingIOError:
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️ Error decoding JSON: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return []