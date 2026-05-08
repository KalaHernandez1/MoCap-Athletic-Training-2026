import json
import numpy as np


# =========================================================
# USER CONFIG
# =========================================================

MANIFEST_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/pose_outputs/manifest.json"
EXTRINSICS_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/extrinsics_robotpy_result.json"

OUTPUT_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/reconstruction_3d_master_slave2.json"

CAM_A = "master"
CAM_B = "slave2"

MIN_VISIBILITY = 0.5
MAX_REPROJ_ERROR_PX = 12.0

STRICT_NULL_ON_BAD_POINTS = True


# =========================================================
# MEDIAPIPE POSE LANDMARK IDS
# =========================================================

LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

ANGLE_DEFS = {
    "left_elbow_angle": (11, 13, 15),
    "right_elbow_angle": (12, 14, 16),
    "left_knee_angle": (23, 25, 27),
    "right_knee_angle": (24, 26, 28),
    "left_shoulder_angle": (13, 11, 23),
    "right_shoulder_angle": (14, 12, 24),
    "left_hip_angle": (11, 23, 25),
    "right_hip_angle": (12, 24, 26),
}


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_pose_json_paths_from_manifest(manifest):
    cams = manifest["cameras"]
    return {
        "master": cams["cam1"]["pose_json"],
        "slave2": cams["cam3"]["pose_json"],
    }


def extract_projection_matrices(extrinsics_data):
    return {
        "master": np.array(extrinsics_data["P_master"], dtype=np.float64),
        "slave2": np.array(extrinsics_data["P_slave2"], dtype=np.float64),
    }


def get_frame_landmark(frame_data, landmark_idx):
    landmarks = frame_data.get("landmarks", [])
    if landmark_idx < 0 or landmark_idx >= len(landmarks):
        return None

    lm = landmarks[landmark_idx]
    return {
        "u": float(lm["u"]),
        "v": float(lm["v"]),
        "visibility": float(lm.get("visibility", 0.0)),
    }


def triangulate_two_view(P1, P2, pt1, pt2):
    u1, v1 = pt1
    u2, v2 = pt2

    A = np.array([
        u1 * P1[2, :] - P1[0, :],
        v1 * P1[2, :] - P1[1, :],
        u2 * P2[2, :] - P2[0, :],
        v2 * P2[2, :] - P2[1, :]
    ], dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    if abs(X[3]) < 1e-12:
        return None

    X = X / X[3]
    return X[:3]


def reproject_point(P, X):
    X_h = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    x = P @ X_h

    if abs(x[2]) < 1e-12:
        return None

    x = x / x[2]
    return x[:2]


def point_error(p_obs, p_proj):
    return float(np.linalg.norm(np.array(p_obs) - np.array(p_proj)))


def safe_mean(values):
    return float(np.mean(values)) if values else None


def safe_point_to_list(X):
    if X is None:
        return None
    return [float(X[0]), float(X[1]), float(X[2])]


def vec_angle_deg(A, B, C):
    if A is None or B is None or C is None:
        return None

    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)

    BA = A - B
    BC = C - B

    n1 = np.linalg.norm(BA)
    n2 = np.linalg.norm(BC)

    if n1 < 1e-9 or n2 < 1e-9:
        return None

    cosang = np.dot(BA, BC) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def compute_angles_from_points(points3d):
    angles = {}
    for angle_name, (a, b, c) in ANGLE_DEFS.items():
        A = points3d.get(a)
        B = points3d.get(b)
        C = points3d.get(c)
        angles[angle_name] = vec_angle_deg(A, B, C)
    return angles


# =========================================================
# MAIN
# =========================================================

def main():
    manifest = load_json(MANIFEST_JSON)
    extrinsics_data = load_json(EXTRINSICS_JSON)

    pose_json_paths = get_pose_json_paths_from_manifest(manifest)
    P_all = extract_projection_matrices(extrinsics_data)

    P1 = P_all[CAM_A]
    P2 = P_all[CAM_B]

    camA_data = load_json(pose_json_paths[CAM_A])
    camB_data = load_json(pose_json_paths[CAM_B])

    framesA = camA_data["frames"]
    framesB = camB_data["frames"]
    n = min(len(framesA), len(framesB))

    output = {
        "config": {
            "manifest_json": MANIFEST_JSON,
            "extrinsics_json": EXTRINSICS_JSON,
            "output_json": OUTPUT_JSON,
            "camera_pair": [CAM_A, CAM_B],
            "min_visibility": MIN_VISIBILITY,
            "max_reproj_error_px": MAX_REPROJ_ERROR_PX,
            "strict_null_on_bad_points": STRICT_NULL_ON_BAD_POINTS,
        },
        "num_frames": n,
        "frames": [],
        "summary": {}
    }

    total_points = 0
    good_points = 0
    reproj_errors = []
    frame_valid_counts = []

    for frame_idx in range(n):
        fA = framesA[frame_idx]
        fB = framesB[frame_idx]

        points3d_by_idx = {}
        landmark_records = []

        valid_count_this_frame = 0

        for landmark_idx in range(33):
            total_points += 1

            lmA = get_frame_landmark(fA, landmark_idx)
            lmB = get_frame_landmark(fB, landmark_idx)

            rec = {
                "index": landmark_idx,
                "name": LANDMARK_NAMES.get(landmark_idx, f"landmark_{landmark_idx}"),
                "point_3d": None,
                "status": None,
                "master": None,
                "slave2": None,
                "mean_reprojection_error_px": None,
            }

            if lmA is None or lmB is None:
                rec["status"] = "missing_landmark"
                landmark_records.append(rec)
                points3d_by_idx[landmark_idx] = None
                continue

            rec["master"] = {
                "u": lmA["u"],
                "v": lmA["v"],
                "visibility": lmA["visibility"],
            }
            rec["slave2"] = {
                "u": lmB["u"],
                "v": lmB["v"],
                "visibility": lmB["visibility"],
            }

            if lmA["visibility"] < MIN_VISIBILITY or lmB["visibility"] < MIN_VISIBILITY:
                rec["status"] = "low_visibility"
                landmark_records.append(rec)
                points3d_by_idx[landmark_idx] = None
                continue

            ptA = (lmA["u"], lmA["v"])
            ptB = (lmB["u"], lmB["v"])

            X = triangulate_two_view(P1, P2, ptA, ptB)
            if X is None:
                rec["status"] = "triangulation_failed"
                landmark_records.append(rec)
                points3d_by_idx[landmark_idx] = None
                continue

            projA = reproject_point(P1, X)
            projB = reproject_point(P2, X)
            if projA is None or projB is None:
                rec["status"] = "reprojection_failed"
                landmark_records.append(rec)
                points3d_by_idx[landmark_idx] = None
                continue

            errA = point_error(ptA, projA)
            errB = point_error(ptB, projB)
            mean_err = (errA + errB) / 2.0

            rec["master"]["reprojected_u"] = float(projA[0])
            rec["master"]["reprojected_v"] = float(projA[1])
            rec["master"]["error_px"] = errA

            rec["slave2"]["reprojected_u"] = float(projB[0])
            rec["slave2"]["reprojected_v"] = float(projB[1])
            rec["slave2"]["error_px"] = errB

            rec["mean_reprojection_error_px"] = mean_err

            if mean_err > MAX_REPROJ_ERROR_PX:
                rec["status"] = "high_reprojection_error"
                if not STRICT_NULL_ON_BAD_POINTS:
                    rec["point_3d"] = safe_point_to_list(X)
                    points3d_by_idx[landmark_idx] = safe_point_to_list(X)
                else:
                    points3d_by_idx[landmark_idx] = None
                landmark_records.append(rec)
                continue

            rec["status"] = "ok"
            rec["point_3d"] = safe_point_to_list(X)
            points3d_by_idx[landmark_idx] = safe_point_to_list(X)

            good_points += 1
            valid_count_this_frame += 1
            reproj_errors.append(mean_err)

            landmark_records.append(rec)

        frame_valid_counts.append(valid_count_this_frame)

        angles = compute_angles_from_points(points3d_by_idx)

        output["frames"].append({
            "frame_index": frame_idx,
            "image_name_master": fA.get("image_name"),
            "image_name_slave2": fB.get("image_name"),
            "num_valid_3d_landmarks": valid_count_this_frame,
            "landmarks_3d": landmark_records,
            "angles_deg": angles,
        })

    output["summary"] = {
        "total_landmark_attempts": total_points,
        "total_valid_3d_landmarks": good_points,
        "overall_valid_rate": good_points / total_points if total_points > 0 else 0.0,
        "mean_reprojection_error_px": safe_mean(reproj_errors),
        "mean_valid_3d_landmarks_per_frame": safe_mean(frame_valid_counts),
    }

    save_json(OUTPUT_JSON, output)

    print("Saved to:", OUTPUT_JSON)
    print("Frames:", n)
    print("Total landmark attempts:", total_points)
    print("Total valid 3D landmarks:", good_points)
    print("Overall valid rate:", output["summary"]["overall_valid_rate"])
    print("Mean reprojection error (valid points only):", output["summary"]["mean_reprojection_error_px"])
    print("Mean valid 3D landmarks per frame:", output["summary"]["mean_valid_3d_landmarks_per_frame"])


if __name__ == "__main__":
    main()