#!/usr/bin/env python3
"""
Slave Pi Capture Script
=======================
- Runs automatically on boot as a systemd service
- Listens for commands from master Pi over TCP socket
- On receiving 'prepare:<exposure>:<session_num>:<movement_name>':
    * sets exposure
    * creates session folder
    * drains stale buffers
    * marks itself armed
    * replies 'ready'
- Then waits for PWM trigger from master to begin capture

Camera: InnoMaker U20CAM-9281M (OV9281 Monochrome Global Shutter)
Resolution: 1280x800 @ 75fps (MJPG)
Trigger: External via FSIN+ pin from master Pi GPIO 18 (hardware PWM)
"""

import errno
import fcntl
import logging
import mmap
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Tuple

# ==================== CONFIGURATION ====================
RESOLUTION = (1280, 800)
FPS = 75
DEVICE_PATH = "/dev/video0"
BASE_DIR = os.path.expanduser("~/captures")
NUM_BUFFERS = 8
IDLE_TIMEOUT = 2.0
CAMERA_RETRY_INTERVAL = 5.0
CAMERA_STARTUP_TIMEOUT = 120.0
LISTEN_PORT = 9000
DEFAULT_EXPOSURE = 100
TRIGGER_WAIT_TIMEOUT = 30.0

# ==================== V4L2 CONSTANTS ====================
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_S_PARM = 0xC0CC5616
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_PIX_FMT_MJPEG = 0x47504A4D

BUF_SIZE = 88
BUF_OFF_INDEX = 0
BUF_OFF_TYPE = 4
BUF_OFF_BYTESUSED = 8
BUF_OFF_MEMORY = 60
BUF_OFF_M_OFFSET = 64
BUF_OFF_LENGTH = 72

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("slave_capture")


# ==================== CAMERA SETUP ====================
def set_exposure(device_path: str, exposure: int) -> None:
    cmds = [
        ["v4l2-ctl", "-d", device_path, "--set-ctrl", "focus_automatic_continuous=1"],
        ["v4l2-ctl", "-d", device_path, "--set-ctrl", "auto_exposure=1"],
        ["v4l2-ctl", "-d", device_path, "--set-ctrl", f"exposure_time_absolute={exposure}"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            log.warning("Command failed: %s | stderr=%s", " ".join(cmd), exc.stderr.strip())
    log.info("Exposure set to %s", exposure)



def wait_for_camera() -> bool:
    log.info("Waiting for camera to become available...")
    start_time = time.time()
    while time.time() - start_time < CAMERA_STARTUP_TIMEOUT:
        if os.path.exists(DEVICE_PATH):
            try:
                fd = os.open(DEVICE_PATH, os.O_RDWR)
                os.close(fd)
                log.info("Camera found at %s", DEVICE_PATH)
                return True
            except OSError:
                pass
        log.info("Camera not ready, retrying in %ss...", CAMERA_RETRY_INTERVAL)
        time.sleep(CAMERA_RETRY_INTERVAL)
    log.error("Camera not found after %ss", CAMERA_STARTUP_TIMEOUT)
    return False


# ==================== V4L2 CAMERA ====================
class V4L2Camera:
    def __init__(self, device_path: str, width: int, height: int):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fd: Optional[int] = None
        self.buffers = []

    def open(self) -> None:
        self.fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)

        fmt_buf = bytearray(208)
        struct.pack_into("I", fmt_buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", fmt_buf, 4, self.width)
        struct.pack_into("I", fmt_buf, 8, self.height)
        struct.pack_into("I", fmt_buf, 12, V4L2_PIX_FMT_MJPEG)
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt_buf)
        actual_w = struct.unpack_from("I", fmt_buf, 4)[0]
        actual_h = struct.unpack_from("I", fmt_buf, 8)[0]
        log.info("Format set: %sx%s MJPG", actual_w, actual_h)

        parm = bytearray(204)
        struct.pack_into("I", parm, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", parm, 12, 1)   # numerator
        struct.pack_into("I", parm, 16, 120)  # denominator
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        log.info("Frame rate set to %sfps", FPS)

        reqbuf = bytearray(20)
        struct.pack_into("I", reqbuf, 0, NUM_BUFFERS)
        struct.pack_into("I", reqbuf, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", reqbuf, 8, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbuf)
        count = struct.unpack_from("I", reqbuf, 0)[0]
        log.info("Allocated %s V4L2 buffers", count)

        self.buffers = []
        for i in range(count):
            buf = bytearray(BUF_SIZE)
            struct.pack_into("I", buf, BUF_OFF_INDEX, i)
            struct.pack_into("I", buf, BUF_OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into("I", buf, BUF_OFF_MEMORY, V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)

            length = struct.unpack_from("I", buf, BUF_OFF_LENGTH)[0]
            offset = struct.unpack_from("I", buf, BUF_OFF_M_OFFSET)[0]
            mm = mmap.mmap(
                self.fd,
                length,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset,
            )
            self.buffers.append((mm, length))

            qbuf = bytearray(BUF_SIZE)
            struct.pack_into("I", qbuf, BUF_OFF_INDEX, i)
            struct.pack_into("I", qbuf, BUF_OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into("I", qbuf, BUF_OFF_MEMORY, V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)

        buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type)
        log.info("Streaming started")

    def read_frame_nonblock(self) -> Tuple[bool, Optional[bytes]]:
        buf = bytearray(BUF_SIZE)
        struct.pack_into("I", buf, BUF_OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", buf, BUF_OFF_MEMORY, V4L2_MEMORY_MMAP)
        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        except OSError as e:
            if e.errno == errno.EAGAIN:
                return False, None
            raise

        idx = struct.unpack_from("I", buf, BUF_OFF_INDEX)[0]
        bytesused = struct.unpack_from("I", buf, BUF_OFF_BYTESUSED)[0]

        data = None
        if bytesused > 0:
            mm, _ = self.buffers[idx]
            mm.seek(0)
            data = mm.read(bytesused)

        qbuf = bytearray(BUF_SIZE)
        struct.pack_into("I", qbuf, BUF_OFF_INDEX, idx)
        struct.pack_into("I", qbuf, BUF_OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", qbuf, BUF_OFF_MEMORY, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)

        if bytesused == 0:
            return False, None
        return True, data

    def wait_for_frame(self, timeout: float) -> Tuple[bool, Optional[bytes]]:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return False, None
        return self.read_frame_nonblock()

    def drain(self) -> int:
        count = 0
        while True:
            ok, _ = self.read_frame_nonblock()
            if not ok:
                break
            count += 1
        return count

    def stop(self) -> None:
        if self.fd is not None:
            try:
                buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type)
            except Exception:
                pass
            for mm, _ in self.buffers:
                mm.close()
            os.close(self.fd)
            self.fd = None
        log.info("Streaming stopped")


# ==================== SESSION ====================
def create_session_folder(session_number: int, movement_name: str = "") -> Tuple[str, str]:
    os.makedirs(BASE_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = movement_name.strip().lower().replace(" ", "_") if movement_name else "movement"
    folder_name = f"{safe_name}_session_{session_number:03d}_{timestamp}"
    folder_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, timestamp


# ==================== SOCKET LISTENER ====================
def start_command_server(cam: V4L2Camera, ready_event: threading.Event, prepare_data: Dict) -> None:
    """
    Listen for prepare commands from master.
    Message format: 'prepare:<exposure>:<session_num>:<movement_name>'
    Responds with 'ready' only after exposure is set, folder exists, and stale frames are drained.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", LISTEN_PORT))
    server.listen(5)
    log.info("Command server listening on port %s", LISTEN_PORT)

    while True:
        conn = None
        try:
            conn, addr = server.accept()
            log.info("Connection from %s", addr[0])
            msg = conn.recv(1024).decode().strip()
            log.info("Received: %s", msg)

            if not msg.startswith("prepare:"):
                conn.sendall(b"error:unknown_command\n")
                continue

            parts = msg.split(":", 3)
            if len(parts) < 3:
                conn.sendall(b"error:bad_prepare\n")
                continue

            exposure = int(parts[1])
            session_num = int(parts[2])
            movement_name = parts[3] if len(parts) > 3 else "movement"

            set_exposure(DEVICE_PATH, exposure)
            time.sleep(0.3)

            folder_path, timestamp = create_session_folder(session_num, movement_name)
            drained = cam.drain()
            log.info("Session %03d folder ready - %s", session_num, folder_path)
            log.info("Drained %s stale frames before arming", drained)

            prepare_data.clear()
            prepare_data.update(
                {
                    "folder_path": folder_path,
                    "timestamp": timestamp,
                    "session_num": session_num,
                    "movement_name": movement_name,
                    "exposure": exposure,
                    "armed": True,
                }
            )

            ready_event.set()
            conn.sendall(b"ready\n")

        except Exception as e:
            log.error("Command server error: %s", e)
            if conn is not None:
                try:
                    conn.sendall(f"error:{e}\n".encode())
                except Exception:
                    pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


# ==================== CAPTURE LOOP ====================
def run_capture_loop(cam: V4L2Camera, ready_event: threading.Event, prepare_data: Dict) -> None:
    hostname = os.uname().nodename

    while True:
        log.info("Waiting for prepare command from master...")
        ready_event.wait()
        ready_event.clear()

        if not prepare_data.get("armed"):
            log.warning("Prepare event set without armed state. Ignoring.")
            continue

        folder_path = prepare_data["folder_path"]
        timestamp = prepare_data["timestamp"]
        session_num = prepare_data["session_num"]
        movement_name = prepare_data["movement_name"]
        exposure = prepare_data["exposure"]

        log.info("Session %03d armed - waiting for PWM trigger...", session_num)

        ret, data = cam.wait_for_frame(timeout=TRIGGER_WAIT_TIMEOUT)
        if not ret or not data:
            log.error("Timed out waiting for trigger for session %03d", session_num)
            prepare_data["armed"] = False
            continue

        log.info("Session %03d started - %s", session_num, folder_path)

        frame_count = 1
        first_frame_time = time.perf_counter()
        last_frame_time = first_frame_time

        filepath = os.path.join(folder_path, f"frame_{frame_count:05d}.jpg")
        with open(filepath, "wb") as f:
            f.write(data)

        while True:
            ret, data = cam.read_frame_nonblock()
            if ret and data:
                frame_count += 1
                last_frame_time = time.perf_counter()
                filepath = os.path.join(folder_path, f"frame_{frame_count:05d}.jpg")
                with open(filepath, "wb") as f:
                    f.write(data)
            elif time.perf_counter() - last_frame_time > IDLE_TIMEOUT:
                break
            else:
                time.sleep(0.0005)

        elapsed = max(last_frame_time - first_frame_time, 1e-9)
        log.info("Session %03d complete!", session_num)
        log.info("  Frames: %s", frame_count)
        log.info("  Duration: %.2fs", elapsed)
        log.info("  Effective FPS: %.1f", frame_count / elapsed)

        metadata_path = os.path.join(folder_path, "metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"movement: {movement_name}\n")
            f.write(f"session: {session_num:03d}\n")
            f.write(f"camera: slave ({hostname})\n")
            f.write(f"resolution: {RESOLUTION[0]}x{RESOLUTION[1]}\n")
            f.write(f"target_fps: {FPS}\n")
            f.write(f"exposure: {exposure}\n")
            f.write(f"frames_captured: {frame_count}\n")
            f.write(f"duration: {elapsed:.2f}s\n")
            f.write(f"timestamp: {timestamp}\n")
        log.info("  Metadata saved.")

        prepare_data["armed"] = False
        log.info("")


# ==================== MAIN ====================
def main() -> None:
    hostname = os.uname().nodename

    log.info("=" * 50)
    log.info("  MOTION CAPTURE - SLAVE PI (%s)", hostname)
    log.info("  Auto-start mode (hardware triggered)")
    log.info("=" * 50)
    log.info("")

    if not wait_for_camera():
        log.error("No camera found. Exiting.")
        sys.exit(1)

    set_exposure(DEVICE_PATH, DEFAULT_EXPOSURE)

    cam = V4L2Camera(DEVICE_PATH, RESOLUTION[0], RESOLUTION[1])
    try:
        cam.open()
    except Exception as e:
        log.error("Failed to open camera: %s", e)
        sys.exit(1)

    time.sleep(0.5)
    drained = cam.drain()
    log.info("Drained %s startup frames. Ready.", drained)

    ready_event = threading.Event()
    prepare_data: Dict = {}

    server_thread = threading.Thread(
        target=start_command_server,
        args=(cam, ready_event, prepare_data),
        daemon=True,
    )
    server_thread.start()

    try:
        run_capture_loop(cam, ready_event, prepare_data)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cam.stop()
        log.info("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()