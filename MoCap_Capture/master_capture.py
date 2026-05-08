#!/usr/bin/env python3
"""
Master Pi Capture Script (V2)
=============================
- Prompts for movement name
- Exposure preview loop on MASTER ONLY using free-run mode
- Sends confirmed exposure + movement name + session number to slaves
- Waits for "ready" from all slaves
- Starts 75 Hz PWM trigger on GPIO 18 for synchronized capture
- Stops PWM when user presses ENTER, then drains remaining triggered frames
  until idle timeout so master behavior matches slaves more closely

Camera behavior on this device:
- focus_automatic_continuous=0  -> free-run mode
- focus_automatic_continuous=1  -> external trigger mode (FSIN)
"""

import os
import time
import fcntl
import mmap
import struct
import errno
import select
import socket
import threading
import subprocess
import re

# ==================== CONFIGURATION ====================
FPS = 75
RESOLUTION = (1280, 800)
DEVICE_PATH = "/dev/video0"
BASE_DIR = os.path.expanduser("~/captures")
NUM_BUFFERS = 8
DEFAULT_EXPOSURE = 100
PREVIEW_PATH = "/tmp/mocap_preview.jpg"

PWM_CHIP = "/sys/class/pwm/pwmchip0"
PWM_PERIOD_NS = int(1e9 / FPS)
PWM_DUTY_NS = 100000  # 100 us pulse

SLAVES = [
    "slave1.local",
    "slave2.local",
]
SLAVE_PORT = 9000
SLAVE_TIMEOUT = 10.0

# Keep this at 120 based on your older working script.
# PWM remains the real 75 Hz capture clock.
DRIVER_FPS_HINT = 120

FIRST_FRAME_TIMEOUT = 5.0
IDLE_TIMEOUT = 2.0

session_counter = 0

# ==================== V4L2 CONSTANTS ====================
VIDIOC_S_FMT      = 0xC0D05605
VIDIOC_S_PARM     = 0xC0CC5616
VIDIOC_REQBUFS    = 0xC0145608
VIDIOC_QUERYBUF   = 0xC0585609
VIDIOC_QBUF       = 0xC058560F
VIDIOC_DQBUF      = 0xC0585611
VIDIOC_STREAMON   = 0x40045612
VIDIOC_STREAMOFF  = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_PIX_FMT_MJPEG          = 0x47504A4D

BUF_SIZE          = 88
BUF_OFF_INDEX     = 0
BUF_OFF_TYPE      = 4
BUF_OFF_BYTESUSED = 8
BUF_OFF_MEMORY    = 60
BUF_OFF_M_OFFSET  = 64
BUF_OFF_LENGTH    = 72


# ==================== HARDWARE PWM ====================
def pwm_start() -> None:
    try:
        with open(f"{PWM_CHIP}/unexport", "w") as f:
            f.write("0")
    except Exception:
        pass

    time.sleep(0.1)

    try:
        with open(f"{PWM_CHIP}/export", "w") as f:
            f.write("0")
    except Exception:
        pass

    time.sleep(0.1)

    with open(f"{PWM_CHIP}/pwm0/duty_cycle", "w") as f:
        f.write("0")
    with open(f"{PWM_CHIP}/pwm0/period", "w") as f:
        f.write(str(PWM_PERIOD_NS))
    with open(f"{PWM_CHIP}/pwm0/duty_cycle", "w") as f:
        f.write(str(PWM_DUTY_NS))
    with open(f"{PWM_CHIP}/pwm0/enable", "w") as f:
        f.write("1")

    print(f"[PWM] Started: {FPS}Hz on GPIO 18")


def pwm_stop() -> None:
    try:
        with open(f"{PWM_CHIP}/pwm0/enable", "w") as f:
            f.write("0")
    except Exception:
        pass

    try:
        with open(f"{PWM_CHIP}/unexport", "w") as f:
            f.write("0")
    except Exception:
        pass

    print("[PWM] Stopped")


# ==================== CAMERA CONTROL ====================
def run_v4l2(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(e.stderr.decode(errors="ignore").strip())
        return False


def set_trigger_mode(device_path: str, enabled: bool) -> None:
    value = "1" if enabled else "0"
    cmd = ["v4l2-ctl", "-d", device_path, "--set-ctrl", f"focus_automatic_continuous={value}"]
    ok = run_v4l2(cmd)
    if ok:
        mode = "external trigger mode" if enabled else "free-run preview mode"
        print(f"[INFO] Camera mode set: {mode}")


def set_exposure_only(device_path: str, exposure: int) -> None:
    cmds = [
        ["v4l2-ctl", "-d", device_path, "--set-ctrl", "auto_exposure=1"],
        ["v4l2-ctl", "-d", device_path, "--set-ctrl", f"exposure_time_absolute={exposure}"],
    ]
    for cmd in cmds:
        run_v4l2(cmd)

    print(f"[INFO] Exposure set to {exposure} ({exposure / 10:.0f}ms)")


def configure_master_for_capture(device_path: str, exposure: int) -> None:
    set_trigger_mode(device_path, enabled=True)
    time.sleep(0.2)
    set_exposure_only(device_path, exposure)
    time.sleep(0.2)


# ==================== SLAVE COORDINATION ====================
def sanitize_movement_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip()
    return cleaned or "movement"


def prepare_slaves(exposure: int, session_num: int, movement_name: str) -> bool:
    safe_movement = sanitize_movement_name(movement_name)
    msg = f"prepare:{exposure}:{session_num}:{safe_movement}\n".encode()
    results = {}

    def contact_slave(host: str) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(SLAVE_TIMEOUT)
            s.connect((host, SLAVE_PORT))
            s.sendall(msg)
            response = s.recv(64).decode().strip()
            s.close()
            results[host] = (response == "ready")
            print(f"[SLAVE] {host}: {response}")
        except Exception as e:
            print(f"[SLAVE] {host}: FAILED - {e}")
            results[host] = False

    threads = [threading.Thread(target=contact_slave, args=(h,)) for h in SLAVES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return all(results.get(h, False) for h in SLAVES)


# ==================== V4L2 CAMERA ====================
class V4L2Camera:
    def __init__(self, device_path: str, width: int, height: int):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fd = None
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
        print(f"[INFO] Format set: {actual_w}x{actual_h} MJPG")

        parm = bytearray(204)
        struct.pack_into("I", parm, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", parm, 12, 1)
        struct.pack_into("I", parm, 16, DRIVER_FPS_HINT)
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        print(f"[INFO] Frame rate set to {DRIVER_FPS_HINT}fps")

        reqbuf = bytearray(20)
        struct.pack_into("I", reqbuf, 0, NUM_BUFFERS)
        struct.pack_into("I", reqbuf, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", reqbuf, 8, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbuf)
        count = struct.unpack_from("I", reqbuf, 0)[0]
        print(f"[INFO] Allocated {count} V4L2 buffers")

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
        print("[INFO] Streaming started")

    def read_frame(self):
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

        if bytesused > 0:
            mm, _ = self.buffers[idx]
            mm.seek(0)
            data = mm.read(bytesused)
        else:
            data = None

        qbuf = bytearray(BUF_SIZE)
        struct.pack_into("I", qbuf, BUF_OFF_INDEX, idx)
        struct.pack_into("I", qbuf, BUF_OFF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", qbuf, BUF_OFF_MEMORY, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)

        if bytesused == 0:
            return False, None
        return True, data

    def wait_for_frame(self, timeout: float):
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return False, None
        return self.read_frame()

    def grab_preview_frame(self):
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 1.0)
            if ready:
                ok, data = self.read_frame()
                if ok and data:
                    return data
        return None

    def drain(self) -> int:
        count = 0
        while True:
            ok, _ = self.read_frame()
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

        print("[INFO] Streaming stopped")


# ==================== SESSION ====================
def get_next_session_number() -> int:
    global session_counter
    session_counter += 1
    return session_counter


def create_session_folder(movement_name: str, session_number: int):
    os.makedirs(BASE_DIR, exist_ok=True)
    safe_name = sanitize_movement_name(movement_name).lower().replace(" ", "_")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{safe_name}_session_{session_number:03d}_{timestamp}"
    folder_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, timestamp


# ==================== EXPOSURE PREVIEW ====================
def exposure_preview(cam: V4L2Camera, device_path: str) -> int:
    exposure = DEFAULT_EXPOSURE

    while True:
        val = input(f"\nEnter exposure value (current: {exposure}, press ENTER to keep): ").strip()
        if val:
            try:
                candidate = int(val)
            except ValueError:
                print("[ERROR] Please enter a number.")
                continue

            if not 1 <= candidate <= 133:
                print("[ERROR] Exposure must be between 1 and 133.")
                continue

            exposure = candidate

        set_trigger_mode(device_path, enabled=False)
        time.sleep(0.2)

        set_exposure_only(device_path, exposure)
        time.sleep(0.3)

        cam.drain()

        print("[PREVIEW] Capturing preview frame...")
        data = cam.grab_preview_frame()

        set_trigger_mode(device_path, enabled=True)
        time.sleep(0.2)

        if data is None:
            print("[PREVIEW] Failed to capture frame, try again.")
            continue

        with open(PREVIEW_PATH, "wb") as f:
            f.write(data)

        print(f"[PREVIEW_FILE] {PREVIEW_PATH}", flush=True)

        confirm = input("Happy with exposure? (y to continue, or enter new value): ").strip().lower()
        if confirm == "y":
            return exposure


# ==================== CAPTURE ====================
def capture_frames(cam: V4L2Camera, folder_path: str) -> int:
    frame_count = 0
    pwm_stop_requested = threading.Event()

    def wait_for_stop():
        input()
        pwm_stop_requested.set()

    stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
    stop_thread.start()

    print("[CAPTURE] Recording... Press ENTER to stop PWM.")

    cam.drain()

    # Wait for first triggered frame
    first_deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT
    first_frame = None
    while time.perf_counter() < first_deadline:
        ret, data = cam.wait_for_frame(timeout=0.5)
        if ret and data:
            first_frame = data
            break

    if first_frame is None:
        print("[ERROR] No triggered frames received on master.")
        return 0

    start_time = time.perf_counter()
    last_frame_time = start_time

    frame_count += 1
    filepath = os.path.join(folder_path, f"frame_{frame_count:05d}.jpg")
    with open(filepath, "wb") as f:
        f.write(first_frame)

    pwm_has_been_stopped = False

    while True:
        if pwm_stop_requested.is_set() and not pwm_has_been_stopped:
            pwm_stop()
            pwm_has_been_stopped = True
            print("[CAPTURE] PWM stopped. Draining remaining triggered frames...")

        ret, data = cam.read_frame()
        now = time.perf_counter()

        if ret and data:
            frame_count += 1
            last_frame_time = now
            filepath = os.path.join(folder_path, f"frame_{frame_count:05d}.jpg")
            with open(filepath, "wb") as f:
                f.write(data)
        else:
            time.sleep(0.001)

        if pwm_has_been_stopped and (now - last_frame_time > IDLE_TIMEOUT):
            break

    elapsed = max(last_frame_time - start_time, 1e-6)

    print("\n[CAPTURE] Complete!")
    print(f"[CAPTURE] Frames: {frame_count}")
    print(f"[CAPTURE] Duration: {elapsed:.2f}s")
    print(f"[CAPTURE] Effective FPS: {frame_count / elapsed:.1f}")

    return frame_count


# ==================== MAIN ====================
def main() -> None:
    print("=" * 50)
    print("  MOTION CAPTURE - MASTER PI")
    print("=" * 50)
    print()

    print("[SETUP] Opening camera...")
    cam = V4L2Camera(DEVICE_PATH, RESOLUTION[0], RESOLUTION[1])

    try:
        cam.open()
    except Exception as e:
        print(f"[ERROR] Failed to open camera: {e}")
        return

    configure_master_for_capture(DEVICE_PATH, DEFAULT_EXPOSURE)
    print("[SETUP] Camera ready.")

    try:
        while True:
            print()
            movement = input("Enter movement name (or 'quit' to exit): ").strip()
            if movement.lower() == "quit":
                break
            if not movement:
                print("[ERROR] Movement name cannot be empty.")
                continue

            exposure = exposure_preview(cam, DEVICE_PATH)
            print(f"\n[INFO] Exposure confirmed: {exposure}")

            configure_master_for_capture(DEVICE_PATH, exposure)

            next_session = session_counter + 1

            print(f"\n[SLAVES] Preparing slaves (exposure={exposure}, session={next_session:03d})...")
            all_ready = prepare_slaves(exposure, next_session, movement)

            if not all_ready:
                print("[ERROR] Not all slaves responded. Check connections and try again.")
                continue

            session_num = get_next_session_number()

            print("[SLAVES] All slaves ready.")
            time.sleep(0.5)

            folder_path, timestamp = create_session_folder(movement, session_num)

            print(f"\n[SESSION] Movement:  {movement}")
            print(f"[SESSION] Session:   {session_num:03d}")
            print(f"[SESSION] Timestamp: {timestamp}")
            print(f"[SESSION] Folder:    {folder_path}")
            print()

            input("Press ENTER to start recording...")

            pwm_start()
            frame_count = capture_frames(cam, folder_path)

            metadata_path = os.path.join(folder_path, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(f"movement: {movement}\n")
                f.write(f"session: {session_num:03d}\n")
                f.write("camera: master\n")
                f.write(f"resolution: {RESOLUTION[0]}x{RESOLUTION[1]}\n")
                f.write(f"target_fps: {FPS}\n")
                f.write(f"driver_fps_hint: {DRIVER_FPS_HINT}\n")
                f.write(f"exposure: {exposure}\n")
                f.write(f"frames_captured: {frame_count}\n")
                f.write(f"timestamp: {timestamp}\n")
            print("[SESSION] Metadata saved.")

            cont = input("\nCapture another movement? (y/n): ").strip().lower()
            if cont != "y":
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pwm_stop()
        cam.stop()
        print("[INFO] Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
