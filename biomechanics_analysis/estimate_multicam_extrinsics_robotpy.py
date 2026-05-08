import os
import json
import re
from pathlib import Path

import cv2
import numpy as np
import robotpy_apriltag as apriltag


# =========================================================
# USER CONFIG
# =========================================================

INTRINSICS = {
    "master": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/calibration_master_800x600.npz",
    "slave1": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/calibration_slave1_800x600.npz",
    "slave2": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/calibration_slave2_800x600.npz",
}

FRAME_DIRS = {
    "master": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/test4_cam1_frames",
    "slave1": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/test4_cam2_frames",
    "slave2": "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/test4_cam3_frames",
}

OUTPUT_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/extrinsics_robotpy_result.json"

TAG_FAMILY = "tag36h11"
TAG_ID = 1
TAG_SIZE_M = 0.165

MAX_REPROJ_ERROR_PX = 2.0
USE_MEDIAN_TRANSLATION = True

DEBUG_PRINT_FIRST_N = 10


# =========================================================
# DETECTOR
# =========================================================

def make_detector(family=TAG_FAMILY):
    det = apriltag.AprilTagDetector()
    det.addFamily(family)
    cfg = det.getConfig()
    cfg.quadDecimate = 1.0
    cfg.refineEdges = True
    cfg.numThreads = 4
    det.setConfig(cfg)
    return det


def detection_corners(det_result):
    """
    robotpy_apriltag corners:
    CCW starting from bottom-left
    """
    pts = np.empty((4, 2), dtype=np.float64)
    for i in range(4):
        c = det_result.getCorner(i)
        pts[i, 0] = c.x
        pts[i, 1] = c.y
    return pts


# =========================================================
# IO / GEOMETRY HELPERS
# =========================================================

def load_intrinsics(npz_path):
    d = np.load(npz_path)
    K = d["camera_matrix"].astype(np.float64)
    dist = d["dist_coeffs"].astype(np.float64).reshape(-1)
    res = tuple(d["resolution"].tolist()) if "resolution" in d else None
    reproj = float(d["reprojection_error"]) if "reprojection_error" in d else None
    return K, dist, res, reproj


def tag_object_points(size_m):
    """
    CCW starting from bottom-left,
    matching robotpy_apriltag corner order
    """
    s = size_m / 2.0
    return np.array([
        [-s, -s, 0.0],   # BL
        [ s, -s, 0.0],   # BR
        [ s,  s, 0.0],   # TR
        [-s,  s, 0.0],   # TL
    ], dtype=np.float64)


def solve_tag_pose(corners_2d, K, dist, size_m):
    obj_pts = tag_object_points(size_m)
    img_pts = corners_2d.astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
    reproj_px = float(np.sqrt(np.mean(err ** 2)))

    R, _ = cv2.Rodrigues(rvec)

    T_cam_tag = np.eye(4, dtype=np.float64)
    T_cam_tag[:3, :3] = R
    T_cam_tag[:3, 3] = tvec.reshape(3)

    return {
        "R": R,
        "t": tvec.reshape(3),
        "T_cam_tag": T_cam_tag,
        "reproj_px": reproj_px,
    }


def rotation_angle_deg(R):
    angle_rad = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def transform_to_dict(T):
    return {
        "R": T[:3, :3].tolist(),
        "t": T[:3, 3].tolist(),
        "T_4x4": T.tolist(),
    }


def average_rotations_from_matrices(R_list):
    M = np.zeros((3, 3), dtype=np.float64)
    for R in R_list:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def robust_average_transforms(T_list, use_median_translation=True):
    R_list = [T[:3, :3] for T in T_list]
    t_list = np.array([T[:3, 3] for T in T_list], dtype=np.float64)

    R_avg = average_rotations_from_matrices(R_list)
    if use_median_translation:
        t_avg = np.median(t_list, axis=0)
    else:
        t_avg = np.mean(t_list, axis=0)

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    return T_avg


def projection_matrix(K, T_cam_ref):
    """
    X_cam = R X_ref + t
    P = K [R | t]
    """
    R = T_cam_ref[:3, :3]
    t = T_cam_ref[:3, 3].reshape(3, 1)
    Rt = np.hstack([R, t])
    return K @ Rt


def sorted_frame_paths(folder):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


# =========================================================
# MAIN
# =========================================================

def main():
    detector = make_detector(TAG_FAMILY)

    K_master, dist_master, res_master, calerr_master = load_intrinsics(INTRINSICS["master"])
    K_slave1, dist_slave1, res_slave1, calerr_slave1 = load_intrinsics(INTRINSICS["slave1"])
    K_slave2, dist_slave2, res_slave2, calerr_slave2 = load_intrinsics(INTRINSICS["slave2"])

    master_frames = sorted_frame_paths(FRAME_DIRS["master"])
    slave1_frames = sorted_frame_paths(FRAME_DIRS["slave1"])
    slave2_frames = sorted_frame_paths(FRAME_DIRS["slave2"])

    n = min(len(master_frames), len(slave1_frames), len(slave2_frames))
    print(f"Using {n} synchronized frames.")

    good_T_s1_m = []
    good_T_s2_m = []
    frame_records = []

    stats = {
        "missing_pose_any_view": 0,
        "high_reprojection_error": 0,
        "valid_frames": 0,
    }

    for i in range(n):
        img_m = cv2.imread(str(master_frames[i]), cv2.IMREAD_GRAYSCALE)
        img_s1 = cv2.imread(str(slave1_frames[i]), cv2.IMREAD_GRAYSCALE)
        img_s2 = cv2.imread(str(slave2_frames[i]), cv2.IMREAD_GRAYSCALE)

        if img_m is None or img_s1 is None or img_s2 is None:
            stats["missing_pose_any_view"] += 1
            continue

        if not img_m.flags["C_CONTIGUOUS"]:
            img_m = np.ascontiguousarray(img_m)
        if not img_s1.flags["C_CONTIGUOUS"]:
            img_s1 = np.ascontiguousarray(img_s1)
        if not img_s2.flags["C_CONTIGUOUS"]:
            img_s2 = np.ascontiguousarray(img_s2)

        det_m = [d for d in detector.detect(img_m) if d.getId() == TAG_ID]
        det_s1 = [d for d in detector.detect(img_s1) if d.getId() == TAG_ID]
        det_s2 = [d for d in detector.detect(img_s2) if d.getId() == TAG_ID]

        rec = {
            "frame_index": i,
            "image_master": master_frames[i].name,
            "image_slave1": slave1_frames[i].name,
            "image_slave2": slave2_frames[i].name,
            "used": False,
        }

        if len(det_m) == 0 or len(det_s1) == 0 or len(det_s2) == 0:
            stats["missing_pose_any_view"] += 1
            frame_records.append(rec)
            if i < DEBUG_PRINT_FIRST_N:
                print(f"Frame {i}: missing tag in one or more views")
            continue

        pose_m = solve_tag_pose(detection_corners(det_m[0]), K_master, dist_master, TAG_SIZE_M)
        pose_s1 = solve_tag_pose(detection_corners(det_s1[0]), K_slave1, dist_slave1, TAG_SIZE_M)
        pose_s2 = solve_tag_pose(detection_corners(det_s2[0]), K_slave2, dist_slave2, TAG_SIZE_M)

        if pose_m is None or pose_s1 is None or pose_s2 is None:
            stats["missing_pose_any_view"] += 1
            frame_records.append(rec)
            continue

        rec["reproj_master"] = pose_m["reproj_px"]
        rec["reproj_slave1"] = pose_s1["reproj_px"]
        rec["reproj_slave2"] = pose_s2["reproj_px"]

        if i < DEBUG_PRINT_FIRST_N:
            print(
                f"Frame {i}: reproj = "
                f"{pose_m['reproj_px']:.3f}, "
                f"{pose_s1['reproj_px']:.3f}, "
                f"{pose_s2['reproj_px']:.3f}"
            )

        if (
            pose_m["reproj_px"] > MAX_REPROJ_ERROR_PX or
            pose_s1["reproj_px"] > MAX_REPROJ_ERROR_PX or
            pose_s2["reproj_px"] > MAX_REPROJ_ERROR_PX
        ):
            stats["high_reprojection_error"] += 1
            frame_records.append(rec)
            continue

        T_m_tag = pose_m["T_cam_tag"]
        T_s1_tag = pose_s1["T_cam_tag"]
        T_s2_tag = pose_s2["T_cam_tag"]

        # wanted:
        # X_slave1 = T_slave1_master * X_master
        # X_slave2 = T_slave2_master * X_master
        T_s1_m = T_s1_tag @ np.linalg.inv(T_m_tag)
        T_s2_m = T_s2_tag @ np.linalg.inv(T_m_tag)

        good_T_s1_m.append(T_s1_m)
        good_T_s2_m.append(T_s2_m)

        rec["used"] = True
        rec["T_slave1_master"] = T_s1_m.tolist()
        rec["T_slave2_master"] = T_s2_m.tolist()
        frame_records.append(rec)
        stats["valid_frames"] += 1

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{n}, valid so far: {stats['valid_frames']}")

    result = {
        "config": {
            "intrinsics": INTRINSICS,
            "frame_dirs": FRAME_DIRS,
            "tag_family": TAG_FAMILY,
            "tag_id": TAG_ID,
            "tag_size_m": TAG_SIZE_M,
            "max_reproj_error_px": MAX_REPROJ_ERROR_PX,
            "use_median_translation": USE_MEDIAN_TRANSLATION,
        },
        "intrinsics_summary": {
            "master": {
                "resolution": res_master,
                "calibration_reproj_error": calerr_master,
            },
            "slave1": {
                "resolution": res_slave1,
                "calibration_reproj_error": calerr_slave1,
            },
            "slave2": {
                "resolution": res_slave2,
                "calibration_reproj_error": calerr_slave2,
            },
        },
        "num_total_frames": n,
        "debug_stats": stats,
        "per_frame_records": frame_records,
    }

    print("\n=== DEBUG SUMMARY ===")
    print("missing_pose_any_view =", stats["missing_pose_any_view"])
    print("high_reprojection_error =", stats["high_reprojection_error"])
    print("valid_frames =", stats["valid_frames"])

    if len(good_T_s1_m) >= 5 and len(good_T_s2_m) >= 5:
        T_s1_m_final = robust_average_transforms(
            good_T_s1_m,
            use_median_translation=USE_MEDIAN_TRANSLATION
        )
        T_s2_m_final = robust_average_transforms(
            good_T_s2_m,
            use_median_translation=USE_MEDIAN_TRANSLATION
        )
        T_m_m = np.eye(4, dtype=np.float64)

        P_master = projection_matrix(K_master, T_m_m)
        P_slave1 = projection_matrix(K_slave1, T_s1_m_final)
        P_slave2 = projection_matrix(K_slave2, T_s2_m_final)

        baseline_s1_m = float(np.linalg.norm(T_s1_m_final[:3, 3]))
        baseline_s2_m = float(np.linalg.norm(T_s2_m_final[:3, 3]))

        rotdeg_s1_m = rotation_angle_deg(T_s1_m_final[:3, :3])
        rotdeg_s2_m = rotation_angle_deg(T_s2_m_final[:3, :3])

        result.update({
            "reference_camera": "master",
            "T_master_master": transform_to_dict(T_m_m),
            "T_slave1_master": transform_to_dict(T_s1_m_final),
            "T_slave2_master": transform_to_dict(T_s2_m_final),
            "P_master": P_master.tolist(),
            "P_slave1": P_slave1.tolist(),
            "P_slave2": P_slave2.tolist(),
            "summary": {
                "baseline_slave1_master_m": baseline_s1_m,
                "baseline_slave2_master_m": baseline_s2_m,
                "rotation_slave1_master_deg": rotdeg_s1_m,
                "rotation_slave2_master_deg": rotdeg_s2_m,
            }
        })

        print("\nFinal relative transforms:")
        print("T_slave1_master =")
        print(T_s1_m_final)
        print(f"baseline = {baseline_s1_m:.3f} m, rotation = {rotdeg_s1_m:.2f} deg")

        print("\nT_slave2_master =")
        print(T_s2_m_final)
        print(f"baseline = {baseline_s2_m:.3f} m, rotation = {rotdeg_s2_m:.2f} deg")

    else:
        result["final_status"] = "not_enough_valid_frames_for_final_extrinsics"
        print("\nNot enough valid frames for final extrinsics.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved result to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()