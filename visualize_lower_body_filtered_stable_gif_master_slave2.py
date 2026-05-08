import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# =========================================================
# USER CONFIG
# =========================================================

INPUT_JSON = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/reconstruction_3d_master_slave2.json"

OUTPUT_DIR = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/lower_body_filtered_stable_frames"
OUTPUT_GIF = "/Users/ethan.jiang/Documents/Capstone/mocap_mediapipe/pipeline_test/lower_body_filtered_stable_animation.gif"

START_FRAME = 180
END_FRAME = 295

FPS = 5
FIGSIZE = (7, 7)

# -------------------------
# 仅用于展示的坐标拉伸
# 不改变原始重建数据，只改变画图效果
# -------------------------
DISPLAY_X_SCALE = 3.2
DISPLAY_Y_SCALE = 2.2
DISPLAY_Z_SCALE = 2.8

# -------------------------
# 筛选参数
# -------------------------
MIN_VALID_POINTS = 6       # 5 比较实用；更严格就改 6
MIN_RUN_LENGTH = 3         # 连续片段至少 3 帧

# 骨长一致性阈值（相对中位数）
BONE_REL_TOL = 0.35        # 允许 ±35%
MAX_BAD_BONES = 1          # 最多允许 1 根骨偏差过大

# 时序漂移阈值（单位：米）
JOINT_JUMP_MAX = 0.18      # 单个点最大允许跳变
JOINT_JUMP_MED = 0.10      # 所有公共点位移的中位数
CENTER_JUMP_MAX = 0.12     # 当前帧整体中心跳变阈值
Z_JUMP_MAX = 0.15          # z 方向中心跳变阈值

# 下肢关键点
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
# HELPERS
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
        if len(vals) > 0:
            ref[conn] = float(np.median(vals))
        else:
            ref[conn] = None

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
        return False, {
            "reason": "too_few_bones",
            "checked_bones": checked
        }

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


def is_temporally_stable(cur_pts, prev_pts, frame_gap):
    common_ids = [idx for idx in LOWER_BODY_IDS if cur_pts[idx] is not None and prev_pts[idx] is not None]

    if len(common_ids) < 3:
        return False, {
            "reason": "too_few_common_points",
            "common_points": len(common_ids)
        }

    # gap 越大，适当放宽一点
    scale = min(2.5, 1.0 + 0.25 * max(0, frame_gap - 1))

    joint_dists = []
    for idx in common_ids:
        d = distance(cur_pts[idx], prev_pts[idx])
        joint_dists.append(d)

    max_jump = max(joint_dists)
    med_jump = float(np.median(joint_dists))

    cur_center = get_center(cur_pts)
    prev_center = get_center(prev_pts)

    if cur_center is None or prev_center is None:
        return False, {"reason": "missing_center"}

    center_jump = distance(cur_center, prev_center)
    z_jump = abs(cur_center[2] - prev_center[2])

    if max_jump > JOINT_JUMP_MAX * scale:
        return False, {
            "reason": "max_joint_jump",
            "max_jump": max_jump,
            "threshold": JOINT_JUMP_MAX * scale
        }

    if med_jump > JOINT_JUMP_MED * scale:
        return False, {
            "reason": "median_joint_jump",
            "median_jump": med_jump,
            "threshold": JOINT_JUMP_MED * scale
        }

    if center_jump > CENTER_JUMP_MAX * scale:
        return False, {
            "reason": "center_jump",
            "center_jump": center_jump,
            "threshold": CENTER_JUMP_MAX * scale
        }

    if z_jump > Z_JUMP_MAX * scale:
        return False, {
            "reason": "z_jump",
            "z_jump": z_jump,
            "threshold": Z_JUMP_MAX * scale
        }

    return True, {
        "reason": "ok",
        "common_points": len(common_ids),
        "max_jump": max_jump,
        "median_jump": med_jump,
        "center_jump": center_jump,
        "z_jump": z_jump,
    }


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


def transform_points_for_display(pts):
    """
    只用于显示：
    以 pelvis 中点为中心，对 xyz 做非等比例放大
    """
    center = get_pelvis_center(pts)
    if center is None:
        return pts

    out = {}
    for idx, p in pts.items():
        if p is None:
            out[idx] = None
            continue

        q = p - center
        q = np.array([
            q[0] * DISPLAY_X_SCALE,
            q[1] * DISPLAY_Y_SCALE,
            q[2] * DISPLAY_Z_SCALE,
        ], dtype=np.float64)
        q = q + center
        out[idx] = q

    return out

def collect_axis_limits(frames, kept_indices):
    all_points = []

    for i in kept_indices:
        pts = extract_lower_body_points(frames[i])
        pts = transform_points_for_display(pts)
        for p in pts.values():
            if p is not None:
                all_points.append(p)

    if not all_points:
        return (-1, 1), (-1, 1), (-1, 1)

    arr = np.array(all_points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)

    center = (mins + maxs) / 2.0
    span = np.max(maxs - mins)

    half = max(span * 0.6, 0.25)

    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)

    # z 轴故意拉开一点
    z_half = half * 1.6
    zlim = (center[2] - z_half, center[2] + z_half)

    return xlim, ylim, zlim


def draw_connection(ax, pts, a, b, color):
    pa = pts.get(a)
    pb = pts.get(b)
    if pa is None or pb is None:
        return

    ax.plot(
        [pa[0], pb[0]],
        [pa[1], pb[1]],
        [pa[2], pb[2]],
        linewidth=2.5,
        color=color
    )


def save_frame_plot(frame_index, pts, valid_count, xlim, ylim, zlim, output_path):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # 左腿蓝色
    draw_connection(ax, pts, 23, 25, "tab:blue")
    draw_connection(ax, pts, 25, 27, "tab:blue")

    # 右腿橙色
    draw_connection(ax, pts, 24, 26, "tab:orange")
    draw_connection(ax, pts, 26, 28, "tab:orange")

    # 骨盆绿色
    draw_connection(ax, pts, 23, 24, "tab:green")

    point_colors = {
        23: "tab:blue",
        24: "tab:orange",
        25: "tab:blue",
        26: "tab:orange",
        27: "tab:blue",
        28: "tab:orange",
    }

    for idx in LOWER_BODY_IDS:
        p = pts.get(idx)
        if p is None:
            continue
        ax.scatter(p[0], p[1], p[2], s=45, color=point_colors[idx])
        ax.text(p[0], p[1], p[2], POINT_LABELS[idx], fontsize=9)

    ax.set_title(f"Lower Body 3D | Frame {frame_index} | Valid {valid_count}/6")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect((1, 1, 1.8))
    ax.view_init(elev=18, azim=-65)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    data = load_json(INPUT_JSON)
    frames = data["frames"]

    start_frame = max(0, START_FRAME)
    end_frame = min(END_FRAME, len(frames) - 1)

    # -------------------------------------------------
    # 1) 先按 valid_count 做候选集
    # -------------------------------------------------
    candidate_indices = []
    valid_count_map = {}

    for i in range(start_frame, end_frame + 1):
        pts = extract_lower_body_points(frames[i])
        vc = valid_count_from_pts(pts)
        valid_count_map[i] = vc
        if vc >= MIN_VALID_POINTS:
            candidate_indices.append(i)

    print(f"Range: {start_frame}–{end_frame}")
    print(f"Initial candidate frames (valid_count >= {MIN_VALID_POINTS}): {len(candidate_indices)}")

    if not candidate_indices:
        print("No candidate frames.")
        return

    # -------------------------------------------------
    # 2) 计算参考骨长（中位数）
    # -------------------------------------------------
    ref_bones = compute_reference_bone_lengths(frames, candidate_indices)
    print("\nReference bone lengths:")
    for conn in CONNECTIONS:
        print(f"  {BONE_NAMES[conn]}: {ref_bones[conn]}")

    # -------------------------------------------------
    # 3) 结构筛选 + 时序稳定筛选
    # -------------------------------------------------
    kept_indices = []
    reject_stats = {}

    prev_kept_idx = None
    prev_kept_pts = None

    for i in candidate_indices:
        pts = extract_lower_body_points(frames[i])

        # 结构筛选
        ok_struct, info_struct = is_structurally_good(pts, ref_bones)
        if not ok_struct:
            reason = "struct_" + info_struct["reason"]
            reject_stats[reason] = reject_stats.get(reason, 0) + 1
            continue

        # 第一帧直接保留
        if prev_kept_pts is None:
            kept_indices.append(i)
            prev_kept_idx = i
            prev_kept_pts = pts
            continue

        # 时序稳定筛选
        gap = i - prev_kept_idx
        ok_temp, info_temp = is_temporally_stable(pts, prev_kept_pts, gap)

        if not ok_temp:
            reason = "temp_" + info_temp["reason"]
            reject_stats[reason] = reject_stats.get(reason, 0) + 1
            continue

        kept_indices.append(i)
        prev_kept_idx = i
        prev_kept_pts = pts

    print(f"\nAfter structural + temporal filtering: {len(kept_indices)} frames kept")

    print("\nReject stats:")
    if reject_stats:
        for k, v in sorted(reject_stats.items()):
            print(f"  {k}: {v}")
    else:
        print("  None")

    if not kept_indices:
        print("No frames left after filtering.")
        return

    # -------------------------------------------------
    # 4) 只保留连续片段
    # -------------------------------------------------
    runs = find_runs(kept_indices)

    print("\nDetected runs after stability filtering:")
    final_indices = []
    for run in runs:
        print(f"  {run[0]}–{run[-1]} | length={len(run)}")
        if len(run) >= MIN_RUN_LENGTH:
            final_indices.extend(run)

    print(f"\nAfter run filter (min run length = {MIN_RUN_LENGTH}): {len(final_indices)} frames kept")

    if not final_indices:
        print("No frames left after run filter.")
        return

    print("\nFinal kept frames:")
    print(final_indices)

    # -------------------------------------------------
    # 5) 统一坐标范围并渲染
    # -------------------------------------------------
    xlim, ylim, zlim = collect_axis_limits(frames, final_indices)

    image_paths = []

    for out_idx, frame_idx in enumerate(final_indices):
        pts = extract_lower_body_points(frames[frame_idx])
        vc = valid_count_from_pts(pts)
        pts_display = transform_points_for_display(pts)

        print(f"Render frame {frame_idx}: valid_count = {vc}")

        out_path = os.path.join(
            OUTPUT_DIR,
            f"stable_{out_idx:04d}_orig_{frame_idx:04d}.png"
        )
        save_frame_plot(frame_idx, pts_display, vc, xlim, ylim, zlim, out_path)
        image_paths.append(out_path)

    images = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(OUTPUT_GIF, images, fps=FPS)

    print(f"\nSaved {len(image_paths)} PNG frames to: {OUTPUT_DIR}")
    print(f"Saved GIF to: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()