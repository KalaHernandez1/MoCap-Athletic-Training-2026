# select video & movement
# MAIN RUN FILE

import matplotlib.pyplot as plt
from process import process_csv
from visual3D import load_data, animate_3d
import cv2
import os
from core import run_pipeline  

def load_frames(folder):
    frames = []
    files = sorted(os.listdir(folder))

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is not None:
            frames.append(img)

    return frames

print("\n[MAIN] Generating joint data from camera frames...")
cam1 = load_frames("../PushupTest/pushup_session_003_2026-04-18_14-44-02_MASTER")
cam2 = load_frames("../PushupTest/pushup_session_003_2026-04-18_14-44-02_SLAVE1")
cam3 = load_frames("../PushupTest/pushup_session_003_2026-04-18_14-44-02_SLAVE2")

run_pipeline([cam1, cam2, cam3])  


# can csv path so maybe for different movements make different
# files to keep them seperate to analyze.
csv_path = "../output/joint_data.csv"
movement_type = "squat"

# pipeline
knee_angle, angular_velocity, results = process_csv(csv_path, movement_type)

# results
print("\nMovement Analysis Results:")
for key, value in results.items():
    print(f"{key}: {value}")

# plot
plt.figure(figsize=(10,6))

plt.plot(knee_angle, label="Joint Angle (deg)")
plt.plot(angular_velocity, label="Angular Velocity (deg/s)")

plt.xlabel("Frame Number")
plt.ylabel("Degrees / Degrees per Second")
plt.title(f"{movement_type.capitalize()} Biomechanics Analysis")

plt.legend()
plt.grid(True)
plt.show()

if __name__ == "__main__":

    csv_path = "../output/joint_data.csv"
    df = load_data(csv_path)
    animate_3d(df)
