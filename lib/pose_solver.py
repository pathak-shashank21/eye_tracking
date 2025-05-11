# lib/pose_solver.py
import numpy as np

def get_eye_center(landmarks, left_eye_idx, right_eye_idx):
    left_eye = np.mean([landmarks[i][:2] for i in left_eye_idx], axis=0)
    right_eye = np.mean([landmarks[i][:2] for i in right_eye_idx], axis=0)
    return (left_eye + right_eye) / 2

def get_3d_features(face):
    landmarks = np.array(face['landmarks_3d'])
    if landmarks.shape[1] == 3:  # (num_points, 3)
        nose = landmarks[1]
        left_eye = np.mean(landmarks[[33, 133, 160, 159, 158, 157]], axis=0)
        right_eye = np.mean(landmarks[[362, 263, 387, 386, 385, 384]], axis=0)
        center = (left_eye + right_eye) / 2
        return np.concatenate([center, nose])  # [cx, cy, cz, nx, ny, nz]
    else:
        return np.zeros(6)  # Fallback if shape is unexpected
