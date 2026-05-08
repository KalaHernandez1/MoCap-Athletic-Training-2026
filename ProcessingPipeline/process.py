# load csv, extract joints
# call core & movement logic

import pandas as pd
from core import smooth_signal, compute_3d_angle, compute_velocity
from movements import movement_config, analyze_squat, analyze_pushup

def process_csv(csv_path, movement_type, fps=75):

    dt = 1 / fps
    df = pd.read_csv(csv_path)

    if movement_type not in movement_config:
        raise ValueError("Unsupported movement type")

    jointA_id, jointB_id, jointC_id = movement_config[movement_type]["joint_ids"]

# will change with triangulated points
    jointA = df[df['joint'] == jointA_id][['x','y','z']].to_numpy()
    jointB = df[df['joint'] == jointB_id][['x','y','z']].to_numpy()
    jointC = df[df['joint'] == jointC_id][['x','y','z']].to_numpy()

    jointA = smooth_signal(jointA)
    jointB = smooth_signal(jointB)
    jointC = smooth_signal(jointC)

    angle = compute_3d_angle(jointA, jointB, jointC)
    velocity = compute_velocity(angle, dt)

    if movement_type == "squat":
        results = analyze_squat(angle, velocity)
    elif movement_type == "pushup":
        results = analyze_pushup(angle, velocity)

    return angle, velocity, results
