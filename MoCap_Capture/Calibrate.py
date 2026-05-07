#!/usr/bin/env python3
"""
calibrate_pi.py

Intrinsic calibration for the OV9281 camera on a Raspberry Pi Zero, using the
same V4L2 capture pipeline that the mocap system uses (800x600, MJPG). This
matters because intrinsics calibrated at a different resolution or via a
different driver path (e.g. DirectShow on Windows at 1280x800) do not apply to
the mocap data.

Usage (run on each Pi over `ssh -Y luke@<host>.local`):

    python3 calibrate_pi.py

Hold a 9x6 inner-corner checkerboard (23 mm squares) in front of the camera.
The script captures a frame whenever it sees a valid checkerboard, with a
~1 second cooldown between captures. Vary pose, distance, and tilt across
the captures (corners, edges, near, far, rotated). Aim for 20-25 captures.

Press Q in the preview window to finish and write calibration_<hostname>_800x600.npz.
"""

import cv2
import numpy as np
import subprocess
import socket
import time
from pathlib import Path

# ---- Config ---------------------------------------------------------------

DEVICE = "/dev/video0"
WIDTH = 800
HEIGHT = 600
PIXFMT = "MJPG"

CHECKERBOARD = (9, 6)        # inner corners (cols, rows)
SQUARE_SIZE_M = 0.023        # 23 mm

MIN_CAPTURES = 20
TARGET_CAPTURES = 25
CAPTURE_COOLDOWN_S = 1.0

OUTPUT_DIR = Path.home() / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- V4L2 format negotiation ---------------------------------------------

def set_v4l2_format(device, width, height, pixfmt):
    """Set the device format via v4l2-ctl, then read back what was actually negotiated."""
    subprocess.run(
        ["v4l2-ctl", "-d", device,
         f"--set-fmt-video=width={width},height={height},pixelformat={pixfmt}"],
        check=True,
    )
    out = subprocess.check_output(
        ["v4l2-ctl", "-d", device, "--get-fmt-video"],
        text=True,
    )
    # Parse "Width/Height      : 800/600"
    actual_w = actual_h = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Width/Height"):
            wh = line.split(":", 1)[1].strip()
            actual_w, actual_h = (int(x) for x in wh.split("/"))
            break
    if actual_w is None:
        raise RuntimeError("Could not parse width/height from v4l2-ctl output:\n" + out)
    return actual_w, actual_h

# ---- Main -----------------------------------------------------------------

def main():
    hostname = socket.gethostname()
    print(f"[info] hostname = {hostname}")
    print(f"[info] negotiating {WIDTH}x{HEIGHT} {PIXFMT} on {DEVICE} ...")

    actual_w, actual_h = set_v4l2_format(DEVICE, WIDTH, HEIGHT, PIXFMT)
    print(f"[info] driver returned {actual_w}x{actual_h}")
    if (actual_w, actual_h) != (WIDTH, HEIGHT):
        print(f"[warn] driver did not give us {WIDTH}x{HEIGHT}; got {actual_w}x{actual_h}")
        print("[warn] continuing anyway, but mocap data must use the same resolution")

    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {DEVICE}")

    # Object points template: (0,0,0), (1,0,0), ... scaled by square size
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image
    captured_previews = []

    last_capture_t = 0.0
    image_shape = None

    print("[info] starting preview. press Q to stop and calibrate.")
    print(f"[info] need at least {MIN_CAPTURES} captures, target {TARGET_CAPTURES}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            if image_shape is None:
                image_shape = frame.shape[:2]  # (h, w)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            found, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if found:
                # Subpixel refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                cv2.drawChessboardCorners(display, CHECKERBOARD, refined, found)

                now = time.time()
                if now - last_capture_t > CAPTURE_COOLDOWN_S:
                    obj_points.append(objp.copy())
                    img_points.append(refined)
                    last_capture_t = now
                    n = len(obj_points)
                    print(f"[capture] {n}/{TARGET_CAPTURES}")
                    # Flash the frame so the user sees the capture
                    flash = np.full_like(display, 255)
                    cv2.imshow("calibrate", flash)
                    cv2.waitKey(60)

            # HUD
            n = len(obj_points)
            cv2.putText(display, f"captures: {n}/{TARGET_CAPTURES}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if n >= MIN_CAPTURES else (0, 200, 255), 2)
            cv2.putText(display, "press Q to finish",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("calibrate", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    n = len(obj_points)
    if n < MIN_CAPTURES:
        print(f"[error] only {n} captures, need at least {MIN_CAPTURES}. aborting.")
        return

    print(f"[info] running calibrateCamera on {n} views ...")
    h, w = image_shape
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    print(f"[result] RMS reprojection error: {rms:.4f} px")
    print(f"[result] K =\n{K}")
    print(f"[result] dist = {dist.ravel()}")

    out_path = OUTPUT_DIR / f"calibration_{hostname}_{w}x{h}.npz"
    np.savez(
        out_path,
        K=K,
        dist=dist,
        image_size=np.array([w, h]),
        rms=np.array([rms]),
        n_views=np.array([n]),
        hostname=np.array([hostname]),
    )
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()