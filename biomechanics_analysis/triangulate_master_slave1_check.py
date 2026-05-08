import json
import numpy as np


# =========================================================
# USER CONFIG
# =========================================================

MANIFEST_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/pose_outputs/manifest.json"
EXTRINSICS_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/extrinsics_robotpy_result.json"

OUTPUT_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/triangulation_master_slave1_check.json"

# 先只检查比较稳的点
KEYPOINTS_TO_USE = [11, 12, 23, 24]   # shoulders + hips

MIN_VISIBILITY = 0.5
MAX_REPROJ_ERROR_PX = 15.0


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_pose_json_paths_from_manifest(manifest):
    cams = manifest["cameras"]
    return {
        "master": cams["cam1"]["pose_json"],
        "slave1": cams["cam2"]["pose_json"],
    }


def extract_projection_matrices(extrinsics_data):
    P_master = np.array(extrinsics_data["P_master"], dtype=np.float64)
    P_slave1 = np.array(extrinsics_data["P_slave1"], dtype=np.float64)
    return {
        "master": P_master,
        "slave1": P_slave1,
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
    X = X / X[3]
    return X[:3]


def reproject_point(P, X):
    X_h = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    x = P @ X_h
    x = x / x[2]
    return x[:2]


def point_error(p_obs, p_proj):
    return float(np.linalg.norm(np.array(p_obs) - np.array(p_proj)))


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


def safe_mean(values):
    if not values:
        return None
    return float(np.mean(values))


# =========================================================
# MAIN
# =========================================================

def main():
    manifest = load_json(MANIFEST_JSON)
    extrinsics_data = load_json(EXTRINSICS_JSON)

    pose_json_paths = get_pose_json_paths_from_manifest(manifest)
    P = extract_projection_matrices(extrinsics_data)

    master = load_json(pose_json_paths["master"])
    slave1 = load_json(pose_json_paths["slave1"])

    frames1 = master["frames"]
    frames2 = slave1["frames"]

    n = min(len(frames1), len(frames2))

    results = {
        "config": {
            "manifest_json": MANIFEST_JSON,
            "extrinsics_json": EXTRINSICS_JSON,
            "output_json": OUTPUT_JSON,
            "keypoints_to_use": KEYPOINTS_TO_USE,
            "min_visibility": MIN_VISIBILITY,
            "max_reproj_error_px": MAX_REPROJ_ERROR_PX,
            "camera_pair": ["master", "slave1"]
        },
        "num_frames": n,
        "frames": [],
        "summary": {}
    }

    total_attempts = 0
    total_success = 0
    reproj_errors_all = []

    for i in range(n):
        f1 = frames1[i]
        f2 = frames2[i]

        frame_record = {
            "frame_index": i,
            "keypoints": {}
        }

        for kpt_idx in KEYPOINTS_TO_USE:
            total_attempts += 1

            lm1 = get_frame_landmark(f1, kpt_idx)
            lm2 = get_frame_landmark(f2, kpt_idx)

            kpt_record = {
                "status": None
            }

            if lm1 is None or lm2 is None:
                kpt_record["status"] = "missing_landmark"
                frame_record["keypoints"][str(kpt_idx)] = kpt_record
                continue

            if lm1["visibility"] < MIN_VISIBILITY or lm2["visibility"] < MIN_VISIBILITY:
                kpt_record["status"] = "low_visibility"
                kpt_record["master_visibility"] = lm1["visibility"]
                kpt_record["slave1_visibility"] = lm2["visibility"]
                frame_record["keypoints"][str(kpt_idx)] = kpt_record
                continue

            pt1 = (lm1["u"], lm1["v"])
            pt2 = (lm2["u"], lm2["v"])

            X = triangulate_two_view(P["master"], P["slave1"], pt1, pt2)

            proj1 = reproject_point(P["master"], X)
            proj2 = reproject_point(P["slave1"], X)

            err1 = point_error(pt1, proj1)
            err2 = point_error(pt2, proj2)
            mean_err = (err1 + err2) / 2.0

            kpt_record["X_world_like"] = X.tolist()
            kpt_record["master"] = {
                "observed_uv": [pt1[0], pt1[1]],
                "projected_uv": [float(proj1[0]), float(proj1[1])],
                "error_px": err1,
                "visibility": lm1["visibility"],
            }
            kpt_record["slave1"] = {
                "observed_uv": [pt2[0], pt2[1]],
                "projected_uv": [float(proj2[0]), float(proj2[1])],
                "error_px": err2,
                "visibility": lm2["visibility"],
            }
            kpt_record["mean_reprojection_error_px"] = mean_err

            if mean_err > MAX_REPROJ_ERROR_PX:
                kpt_record["status"] = "high_reprojection_error"
            else:
                kpt_record["status"] = "ok"
                total_success += 1
                reproj_errors_all.append(mean_err)

            frame_record["keypoints"][str(kpt_idx)] = kpt_record

        results["frames"].append(frame_record)

    results["summary"] = {
        "total_attempts": total_attempts,
        "total_success": total_success,
        "success_rate": total_success / total_attempts if total_attempts > 0 else 0.0,
        "mean_reprojection_error_px": safe_mean(reproj_errors_all),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved to:", OUTPUT_JSON)
    print("Total attempts:", total_attempts)
    print("Total success:", total_success)
    print("Success rate:", results["summary"]["success_rate"])
    print("Mean reprojection error:", results["summary"]["mean_reprojection_error_px"])


if __name__ == "__main__":
    main()