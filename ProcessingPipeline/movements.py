# movement specifics (jump, run, squat)

import numpy as np

# type of movement
movement_config = {
    "squat": {
        "joint_ids": (23, 25, 27),  # hip, knee, ankle
        "joint_name": "knee"
    },
    "pushup": {
        "joint_ids": (11, 13, 15),  # shoulder, elbow, wrist
        "joint_name": "elbow"
    }
}

# squat
def analyze_squat(angle, velocity):
    results = {}
    results["min_knee_angle"] = np.min(angle)
    results["max_knee_angle"] = np.max(angle)
    results["peak_velocity"] = np.max(np.abs(velocity))
    return results

# pushup
def analyze_pushup(angle, velocity):
    results = {}
    results["min_elbow_angle"] = np.min(angle)
    results["max_elbow_angle"] = np.max(angle)
    results["peak_velocity"] = np.max(np.abs(velocity))
    return results
