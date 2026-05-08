import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# USER CONFIG
# =========================================================

INPUT_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/reconstruction_3d_master_slave2.json"

OUTPUT_DIR = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/biomech_analysis_output"

START_FRAME = 180
END_FRAME = 295

# -------------------------
# 筛选参数（沿用你原来的思路）
# -------------------------
MIN_VALID_POINTS = 6
MIN_RUN_LENGTH = 3

BONE_REL_TOL = 0.35
MAX_BAD_BONES = 1

JOINT_JUMP_MAX = 0.18
JOINT_JUMP_MED = 0.10
CENTER_JUMP_MAX = 0.12
Z_JUMP_MAX = 0.15

LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28]

CONNECTIONS = [
    (23, 24),   # pelvis
    (23, 25),   # left thigh
    (25, 27),   # left shank
    (24, 26),   # right thigh
    (26, 28),   # right shank
]

POINT_LABELS = {
    23: "LHip",
    24: "RHip",
    25: "LKnee",
    26: "RKnee",
    27: "LAnkle",
    28: "RAnkle",
}

BONE_NAMES = {
    (23, 24): "pelvis",
    (23, 25): "left_thigh",
    (25, 27): "left_shank",
    (24, 26): "right_thigh",
    (26, 28): "right_shank",
}


# =========================================================
# BASIC HELPERS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_lower_body_points(frame):
    pts = {idx: None for idx in LOWER_BODY_IDS}

    for lm in frame.get("landmarks_3d", []):
        idx = lm.get("index")
        if idx not in pts:
            continue

        status = lm.get("status")
        pt = lm.get("point_3d")

        if status == "ok" and pt is not None and len(pt) == 3:
            arr = np.array(pt, dtype=np.float64)
            if np.all(np.isfinite(arr)):
                pts[idx] = arr

    return pts


def valid_count_from_pts(pts):
    return sum(p is not None for p in pts.values())


def distance(a, b):
    return float(np.linalg.norm(a - b))


def compute_bone_lengths(pts):
    lengths = {}
    for a, b in CONNECTIONS:
        pa = pts.get(a)
        pb = pts.get(b)
        if pa is not None and pb is not None:
            lengths[(a, b)] = distance(pa, pb)
        else:
            lengths[(a, b)] = None
    return lengths


def compute_reference_bone_lengths(frames, indices):
    bone_values = {conn: [] for conn in CONNECTIONS}

    for i in indices:
        pts = extract_lower_body_points(frames[i])
        lengths = compute_bone_lengths(pts)
        for conn, val in lengths.items():
            if val is not None:
                bone_values[conn].append(val)

    ref = {}
    for conn, vals in bone_values.items():
        ref[conn] = float(np.median(vals)) if len(vals) > 0 else None

    return ref


def is_structurally_good(pts, ref_bones):
    lengths = compute_bone_lengths(pts)

    bad_bones = 0
    checked = 0
    rel_errors = []

    for conn, cur_len in lengths.items():
        ref_len = ref_bones.get(conn)

        if cur_len is None or ref_len is None or ref_len < 1e-8:
            continue

        checked += 1
        rel_err = abs(cur_len - ref_len) / ref_len
        rel_errors.append(rel_err)

        if rel_err > BONE_REL_TOL:
            bad_bones += 1

    if checked < 3:
        return False, {"reason": "too_few_bones"}

    if bad_bones > MAX_BAD_BONES:
        return False, {
            "reason": "bone_length_outlier",
            "checked_bones": checked,
            "bad_bones": bad_bones,
            "median_rel_error": float(np.median(rel_errors)) if rel_errors else None
        }

    return True, {
        "reason": "ok",
        "checked_bones": checked,
        "bad_bones": bad_bones,
        "median_rel_error": float(np.median(rel_errors)) if rel_errors else None
    }


def get_center(pts):
    arr = [p for p in pts.values() if p is not None]
    if not arr:
        return None
    arr = np.array(arr)
    return arr.mean(axis=0)


def get_pelvis_center(pts):
    lh = pts.get(23)
    rh = pts.get(24)

    valid = []
    if lh is not None:
        valid.append(lh)
    if rh is not None:
        valid.append(rh)

    if len(valid) == 2:
        return (valid[0] + valid[1]) / 2.0
    if len(valid) == 1:
        return valid[0]
    return None


def is_temporally_stable(cur_pts, prev_pts, frame_gap):
    common_ids = [idx for idx in LOWER_BODY_IDS if cur_pts[idx] is not None and prev_pts[idx] is not None]

    if len(common_ids) < 3:
        return False, {"reason": "too_few_common_points"}

    scale = min(2.5, 1.0 + 0.25 * max(0, frame_gap - 1))

    joint_dists = [distance(cur_pts[idx], prev_pts[idx]) for idx in common_ids]

    max_jump = max(joint_dists)
    med_jump = float(np.median(joint_dists))

    cur_center = get_center(cur_pts)
    prev_center = get_center(prev_pts)

    if cur_center is None or prev_center is None:
        return False, {"reason": "missing_center"}

    center_jump = distance(cur_center, prev_center)
    z_jump = abs(cur_center[2] - prev_center[2])

    if max_jump > JOINT_JUMP_MAX * scale:
        return False, {"reason": "max_joint_jump"}
    if med_jump > JOINT_JUMP_MED * scale:
        return False, {"reason": "median_joint_jump"}
    if center_jump > CENTER_JUMP_MAX * scale:
        return False, {"reason": "center_jump"}
    if z_jump > Z_JUMP_MAX * scale:
        return False, {"reason": "z_jump"}

    return True, {"reason": "ok"}


def find_runs(indices):
    if not indices:
        return []

    runs = []
    current = [indices[0]]

    for x in indices[1:]:
        if x == current[-1] + 1:
            current.append(x)
        else:
            runs.append(current)
            current = [x]

    runs.append(current)
    return runs


# =========================================================
# BIOMECH HELPERS
# =========================================================

def unit_vector(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


def angle_between(v1, v2):
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    if u1 is None or u2 is None:
        return None
    c = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def compute_joint_angle(a, b, c):
    """
    计算 ∠ABC
    """
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    return angle_between(ba, bc)


def safe_append(d, key, value):
    d[key].append(np.nan if value is None else value)


def smooth_nan_array(arr, window=5):
    """
    简单移动平均，忽略 nan
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full_like(arr, np.nan)

    half = window // 2
    for i in range(len(arr)):
        s = max(0, i - half)
        e = min(len(arr), i + half + 1)
        chunk = arr[s:e]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            out[i] = np.mean(valid)

    return out


def first_derivative(y):
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)

    for i in range(1, len(y)):
        if np.isfinite(y[i]) and np.isfinite(y[i - 1]):
            out[i] = y[i] - y[i - 1]

    return out


# =========================================================
# FILTER FRAMES
# =========================================================

def get_final_kept_indices(frames, start_frame, end_frame):
    candidate_indices = []
    valid_count_map = {}

    for i in range(start_frame, end_frame + 1):
        pts = extract_lower_body_points(frames[i])
        vc = valid_count_from_pts(pts)
        valid_count_map[i] = vc
        if vc >= MIN_VALID_POINTS:
            candidate_indices.append(i)

    if not candidate_indices:
        return [], valid_count_map, None

    ref_bones = compute_reference_bone_lengths(frames, candidate_indices)

    kept_indices = []
    prev_kept_idx = None
    prev_kept_pts = None

    for i in candidate_indices:
        pts = extract_lower_body_points(frames[i])

        ok_struct, _ = is_structurally_good(pts, ref_bones)
        if not ok_struct:
            continue

        if prev_kept_pts is None:
            kept_indices.append(i)
            prev_kept_idx = i
            prev_kept_pts = pts
            continue

        gap = i - prev_kept_idx
        ok_temp, _ = is_temporally_stable(pts, prev_kept_pts, gap)
        if not ok_temp:
            continue

        kept_indices.append(i)
        prev_kept_idx = i
        prev_kept_pts = pts

    runs = find_runs(kept_indices)

    final_indices = []
    for run in runs:
        if len(run) >= MIN_RUN_LENGTH:
            final_indices.extend(run)

    return final_indices, valid_count_map, ref_bones


# =========================================================
# BUILD ANALYSIS TABLES
# =========================================================

def build_biomech_series(frames, final_indices):
    series = {
        "frame": [],
        "pelvis_x": [],
        "pelvis_y": [],
        "pelvis_z": [],
        "left_knee_angle": [],
        "right_knee_angle": [],
        "left_hip_angle": [],
        "right_hip_angle": [],
        "left_ankle_x": [],
        "left_ankle_y": [],
        "left_ankle_z": [],
        "right_ankle_x": [],
        "right_ankle_y": [],
        "right_ankle_z": [],
        "pelvis_length": [],
        "left_thigh_length": [],
        "left_shank_length": [],
        "right_thigh_length": [],
        "right_shank_length": [],
    }

    for frame_idx in final_indices:
        pts = extract_lower_body_points(frames[frame_idx])

        lhip = pts.get(23)
        rhip = pts.get(24)
        lknee = pts.get(25)
        rknee = pts.get(26)
        lankle = pts.get(27)
        rankle = pts.get(28)

        pelvis = get_pelvis_center(pts)

        # 膝角
        left_knee_angle = compute_joint_angle(lhip, lknee, lankle)
        right_knee_angle = compute_joint_angle(rhip, rknee, rankle)

        # 髋角
        # 用 pelvis center - hip - knee 近似
        left_hip_angle = compute_joint_angle(pelvis, lhip, lknee) if pelvis is not None else None
        right_hip_angle = compute_joint_angle(pelvis, rhip, rknee) if pelvis is not None else None

        lengths = compute_bone_lengths(pts)

        series["frame"].append(frame_idx)

        if pelvis is None:
            safe_append(series, "pelvis_x", None)
            safe_append(series, "pelvis_y", None)
            safe_append(series, "pelvis_z", None)
        else:
            safe_append(series, "pelvis_x", pelvis[0])
            safe_append(series, "pelvis_y", pelvis[1])
            safe_append(series, "pelvis_z", pelvis[2])

        safe_append(series, "left_knee_angle", left_knee_angle)
        safe_append(series, "right_knee_angle", right_knee_angle)
        safe_append(series, "left_hip_angle", left_hip_angle)
        safe_append(series, "right_hip_angle", right_hip_angle)

        if lankle is None:
            safe_append(series, "left_ankle_x", None)
            safe_append(series, "left_ankle_y", None)
            safe_append(series, "left_ankle_z", None)
        else:
            safe_append(series, "left_ankle_x", lankle[0])
            safe_append(series, "left_ankle_y", lankle[1])
            safe_append(series, "left_ankle_z", lankle[2])

        if rankle is None:
            safe_append(series, "right_ankle_x", None)
            safe_append(series, "right_ankle_y", None)
            safe_append(series, "right_ankle_z", None)
        else:
            safe_append(series, "right_ankle_x", rankle[0])
            safe_append(series, "right_ankle_y", rankle[1])
            safe_append(series, "right_ankle_z", rankle[2])

        safe_append(series, "pelvis_length", lengths[(23, 24)])
        safe_append(series, "left_thigh_length", lengths[(23, 25)])
        safe_append(series, "left_shank_length", lengths[(25, 27)])
        safe_append(series, "right_thigh_length", lengths[(24, 26)])
        safe_append(series, "right_shank_length", lengths[(26, 28)])

    return series


# =========================================================
# EVENT DETECTION
# =========================================================

def detect_key_events(series):
    events = {}

    pelvis_z = np.asarray(series["pelvis_z"], dtype=np.float64)
    left_knee = np.asarray(series["left_knee_angle"], dtype=np.float64)
    right_knee = np.asarray(series["right_knee_angle"], dtype=np.float64)
    frames = np.asarray(series["frame"], dtype=np.int32)

    pelvis_z_s = smooth_nan_array(pelvis_z, window=5)
    left_knee_s = smooth_nan_array(left_knee, window=5)
    right_knee_s = smooth_nan_array(right_knee, window=5)
    avg_knee_s = np.nanmean(np.vstack([left_knee_s, right_knee_s]), axis=0)

    if np.any(np.isfinite(pelvis_z_s)):
        dip_idx = int(np.nanargmin(pelvis_z_s))
        events["deepest_pelvis"] = {
            "idx": dip_idx,
            "frame": int(frames[dip_idx]),
            "value": float(pelvis_z_s[dip_idx])
        }

    if np.any(np.isfinite(avg_knee_s)):
        flex_idx = int(np.nanargmin(avg_knee_s))
        ext_idx = int(np.nanargmax(avg_knee_s))
        events["max_knee_flexion"] = {
            "idx": flex_idx,
            "frame": int(frames[flex_idx]),
            "value": float(avg_knee_s[flex_idx])
        }
        events["max_knee_extension"] = {
            "idx": ext_idx,
            "frame": int(frames[ext_idx]),
            "value": float(avg_knee_s[ext_idx])
        }

    return events


# =========================================================
# PLOTTING
# =========================================================

def add_event_line(ax, events, event_name, label, color, y_frac=0.95, x_offset=0.0, ha="left"):
    if event_name not in events:
        return

    x = events[event_name]["frame"]

    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * y_frac

    ax.axvline(x=x, linestyle="--", linewidth=1.2, color=color, alpha=0.8)

    ax.text(
        x + x_offset,
        y,
        label,
        color=color,
        fontsize=9,
        va="top",
        ha=ha,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)
    )

def plot_joint_angles(series, events, output_dir):
    frames = np.asarray(series["frame"])
    left_knee = smooth_nan_array(series["left_knee_angle"], window=5)
    right_knee = smooth_nan_array(series["right_knee_angle"], window=5)
    left_hip = smooth_nan_array(series["left_hip_angle"], window=5)
    right_hip = smooth_nan_array(series["right_hip_angle"], window=5)

    plt.figure(figsize=(11, 6))
    plt.plot(frames, left_knee, label="Left Knee Angle")
    plt.plot(frames, right_knee, label="Right Knee Angle")
    plt.plot(frames, left_hip, label="Left Hip Angle", alpha=0.8)
    plt.plot(frames, right_hip, label="Right Hip Angle", alpha=0.8)

    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.title("Lower Body Joint Angles Over Time", pad=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    add_event_line(ax, events, "deepest_pelvis", "Deepest pelvis", "red", y_frac=0.97, x_offset=0.2, ha="left")
    add_event_line(ax, events, "max_knee_flexion", "Max knee flexion", "purple", y_frac=0.92, x_offset=0.8, ha="left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_angles.png"), dpi=180)
    plt.close()


def plot_pelvis_vertical(series, events, output_dir):
    frames = np.asarray(series["frame"])
    pelvis_z = smooth_nan_array(series["pelvis_z"], window=5)
    dz = first_derivative(pelvis_z)

    plt.figure(figsize=(11, 5))
    plt.plot(frames, pelvis_z, label="Pelvis Z")
    plt.plot(frames, dz, label="Pelvis Z Velocity (frame-to-frame)", alpha=0.8)

    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Pelvis Vertical Motion")
    plt.legend()
    plt.grid(True, alpha=0.3)

    add_event_line(plt.gca(), events, "deepest_pelvis", "Lowest pelvis", "red")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pelvis_vertical_motion.png"), dpi=180)
    plt.close()


def plot_ankle_trajectories(series, output_dir):
    lx = smooth_nan_array(series["left_ankle_x"], window=5)
    ly = smooth_nan_array(series["left_ankle_y"], window=5)
    lz = smooth_nan_array(series["left_ankle_z"], window=5)

    rx = smooth_nan_array(series["right_ankle_x"], window=5)
    ry = smooth_nan_array(series["right_ankle_y"], window=5)
    rz = smooth_nan_array(series["right_ankle_z"], window=5)

    # X-Z view
    plt.figure(figsize=(7, 7))
    plt.plot(lx, lz, label="Left Ankle Trajectory")
    plt.plot(rx, rz, label="Right Ankle Trajectory")
    plt.scatter(lx[0], lz[0], s=40, label="Left Start")
    plt.scatter(rx[0], rz[0], s=40, label="Right Start")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Ankle Trajectory (X-Z View)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ankle_trajectory_xz.png"), dpi=180)
    plt.close()

    # Y-Z view
    plt.figure(figsize=(7, 7))
    plt.plot(ly, lz, label="Left Ankle Trajectory")
    plt.plot(ry, rz, label="Right Ankle Trajectory")
    plt.scatter(ly[0], lz[0], s=40, label="Left Start")
    plt.scatter(ry[0], rz[0], s=40, label="Right Start")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("Ankle Trajectory (Y-Z View)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ankle_trajectory_yz.png"), dpi=180)
    plt.close()


def plot_bone_length_consistency(series, output_dir):
    frames = np.asarray(series["frame"])

    pelvis = smooth_nan_array(series["pelvis_length"], window=3)
    lt = smooth_nan_array(series["left_thigh_length"], window=3)
    ls = smooth_nan_array(series["left_shank_length"], window=3)
    rt = smooth_nan_array(series["right_thigh_length"], window=3)
    rs = smooth_nan_array(series["right_shank_length"], window=3)

    plt.figure(figsize=(11, 6))
    plt.plot(frames, pelvis, label="Pelvis")
    plt.plot(frames, lt, label="Left Thigh")
    plt.plot(frames, ls, label="Left Shank")
    plt.plot(frames, rt, label="Right Thigh")
    plt.plot(frames, rs, label="Right Shank")

    plt.xlabel("Frame")
    plt.ylabel("Length (m)")
    plt.title("Bone Length Consistency Check")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bone_length_consistency.png"), dpi=180)
    plt.close()


def plot_summary_dashboard(series, events, output_dir):
    frames = np.asarray(series["frame"])

    left_knee = smooth_nan_array(series["left_knee_angle"], window=5)
    right_knee = smooth_nan_array(series["right_knee_angle"], window=5)
    pelvis_z = smooth_nan_array(series["pelvis_z"], window=5)

    lx = smooth_nan_array(series["left_ankle_x"], window=5)
    lz = smooth_nan_array(series["left_ankle_z"], window=5)
    rx = smooth_nan_array(series["right_ankle_x"], window=5)
    rz = smooth_nan_array(series["right_ankle_z"], window=5)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(frames, left_knee, label="Left Knee")
    ax1.plot(frames, right_knee, label="Right Knee")
    ax1.set_title("Knee Angles")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("deg")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frames, pelvis_z, label="Pelvis Z")
    ax2.set_title("Pelvis Vertical Motion")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Z")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(lx, lz, label="Left Ankle")
    ax3.plot(rx, rz, label="Right Ankle")
    ax3.set_title("Ankle Trajectory (X-Z)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis("equal")

    ax4 = fig.add_subplot(2, 2, 4)
    text_lines = []
    if "deepest_pelvis" in events:
        text_lines.append(f"Deepest pelvis frame: {events['deepest_pelvis']['frame']}")
    if "max_knee_flexion" in events:
        text_lines.append(f"Max knee flexion frame: {events['max_knee_flexion']['frame']}")
        text_lines.append(f"Min avg knee angle: {events['max_knee_flexion']['value']:.2f} deg")
    if "max_knee_extension" in events:
        text_lines.append(f"Max knee extension frame: {events['max_knee_extension']['frame']}")
        text_lines.append(f"Max avg knee angle: {events['max_knee_extension']['value']:.2f} deg")

    text_lines.append(f"Total analyzed frames: {len(frames)}")

    ax4.axis("off")
    ax4.text(
        0.02, 0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biomech_dashboard.png"), dpi=180)
    plt.close()


# =========================================================
# SAVE SUMMARY TXT
# =========================================================

def write_summary(series, events, ref_bones, output_dir):
    lines = []
    frames = series["frame"]

    lines.append("Biomechanics Analysis Summary")
    lines.append("=" * 40)
    lines.append(f"Analyzed frame count: {len(frames)}")
    if len(frames) > 0:
        lines.append(f"Frame range used: {frames[0]} - {frames[-1]}")
    lines.append("")

    lines.append("Reference bone lengths (median):")
    if ref_bones is not None:
        for conn in CONNECTIONS:
            name = BONE_NAMES[conn]
            val = ref_bones[conn]
            if val is None:
                lines.append(f"  {name}: None")
            else:
                lines.append(f"  {name}: {val:.5f} m")
    lines.append("")

    lk = smooth_nan_array(series["left_knee_angle"], window=5)
    rk = smooth_nan_array(series["right_knee_angle"], window=5)
    pz = smooth_nan_array(series["pelvis_z"], window=5)

    def stat_line(name, arr):
        valid = np.asarray(arr)[np.isfinite(arr)]
        if len(valid) == 0:
            return f"{name}: no valid values"
        return f"{name}: mean={np.mean(valid):.2f}, min={np.min(valid):.2f}, max={np.max(valid):.2f}"

    lines.append("Key metrics:")
    lines.append("  " + stat_line("Left knee angle", lk))
    lines.append("  " + stat_line("Right knee angle", rk))
    lines.append("  " + stat_line("Pelvis Z", pz))
    lines.append("")

    lines.append("Detected events:")
    if not events:
        lines.append("  None")
    else:
        for k, v in events.items():
            lines.append(f"  {k}: frame={v['frame']}, value={v['value']:.5f}")

    with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
        f.write("\n".join(lines))


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    data = load_json(INPUT_JSON)
    frames = data["frames"]

    start_frame = max(0, START_FRAME)
    end_frame = min(END_FRAME, len(frames) - 1)

    final_indices, valid_count_map, ref_bones = get_final_kept_indices(frames, start_frame, end_frame)

    print(f"Frame range: {start_frame} - {end_frame}")
    print(f"Final kept frames: {len(final_indices)}")

    if not final_indices:
        print("No valid stable frames found.")
        return

    series = build_biomech_series(frames, final_indices)
    events = detect_key_events(series)

    plot_joint_angles(series, events, OUTPUT_DIR)
    plot_pelvis_vertical(series, events, OUTPUT_DIR)
    plot_ankle_trajectories(series, OUTPUT_DIR)
    plot_bone_length_consistency(series, OUTPUT_DIR)
    plot_summary_dashboard(series, events, OUTPUT_DIR)
    write_summary(series, events, ref_bones, OUTPUT_DIR)

    print("\nSaved analysis results to:")
    print(OUTPUT_DIR)
    print("\nGenerated files:")
    print("  - joint_angles.png")
    print("  - pelvis_vertical_motion.png")
    print("  - ankle_trajectory_xz.png")
    print("  - ankle_trajectory_yz.png")
    print("  - bone_length_consistency.png")
    print("  - biomech_dashboard.png")
    print("  - analysis_summary.txt")


if __name__ == "__main__":
    main()