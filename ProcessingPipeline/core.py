# filtering, 3d angle & velocity calculation

import numpy as np
from scipy.signal import savgol_filter
import process
import csv
import os

# filter
def smooth_signal(data, window_length=11, polyorder=2):
    return savgol_filter(data, window_length, polyorder, axis=0)

# 3d angles
def compute_3d_angle(A, B, C):
    BA = A - B
    BC = C - B

    dot = np.einsum('ij,ij->i', BA, BC)
    norms = np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)

    cosine = dot / np.clip(norms, 1e-8, None)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.arccos(cosine)
    return np.degrees(angle)

# angular velocity
def compute_velocity(angle, dt):
    return np.gradient(angle, dt)

def run_pipeline(camera_frames):
    print("[CORE] Extracting keypoints...")

    keypoints_2d = process.extract_keypoints(camera_frames)

    print("[CORE] Triangulating to 3D...")

    points_3d = process.triangulate(keypoints_2d)

    print("[CORE] Saving to CSV...")

    save_to_csv(points_3d)

    print("[CORE] Done.")


def save_to_csv(points_3d):
    output_path = "../output/joint_data.csv"

    os.makedirs("../output", exist_ok=True)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["frame", "joint", "x", "y", "z"])

        for frame_idx, frame in enumerate(points_3d):
            for joint_idx, joint in enumerate(frame):
                x, y, z = joint

                writer.writerow([frame_idx, joint_idx, x, y, z])


