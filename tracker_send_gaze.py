import os
import cv2
import numpy as np
from input_reader import InputReader
from facetracker import FaceTracker
from gaze_dot import gaze_dot_overlay
from calibrate import calibrate
from multiprocessing import Process, Pipe

CALIBRATION_FILE = "calibration_affine.npy"
MODEL_PATH = "models"

def run_tracker_with_overlay():
    input_reader = InputReader(source=0)
    face_tracker = FaceTracker(MODEL_PATH)

    affine = None
    if os.path.exists(CALIBRATION_FILE):
        affine = np.load(CALIBRATION_FILE, allow_pickle=True)
    if affine is None:
        affine = calibrate(face_tracker, input_reader)
    if affine is None:
        print("⚠️ Skipping gaze tracking due to failed calibration.")
        return

    parent_conn, child_conn = Pipe()
    overlay_proc = Process(target=gaze_dot_overlay, args=(parent_conn,))
    overlay_proc.start()

    try:
        while True:
            frame = input_reader.read()
            if frame is None:
                continue
            detections = face_tracker.detect_faces(frame)
            if not detections:
                continue
            eye = face_tracker.get_eye_center(detections[0])
            if eye is None:
                continue
            eye = np.array([[eye]], dtype=np.float32)
            mapped = cv2.transform(eye, affine)
            x, y = int(mapped[0][0][0]), int(mapped[0][0][1])
            child_conn.send((x, y))

            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass

    child_conn.send("_END_")
    overlay_proc.join()
    input_reader.release()
