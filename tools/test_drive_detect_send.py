#!/usr/bin/env python3
"""
tools/test_drive_detect_send.py — Merged: Auto-Drive + Purple Detect + Depth + STM32 Send
===========================================================================================
Combines ESP32 chassis control with purple detection, stereo depth, and STM32 coordinate sending.

Flow:
  1. No purple detected → ESP32 drives FORWARD automatically
  2. Purple detected    → ESP32 STOPS immediately
                        → Compute stereo depth to the purple object
                        → Send (X, Y, Z) coordinates to STM32 via UART

STM32 coordinate system (X#Y#Z):
  X = horizontal angle: 0=centre, +left, -right  (mapped from pixel cx)
  Y = depth in mm from stereo engine
  Z = height in arm coordinate system (0–325mm, ref 0 is 390mm below camera)

Controls:
  's' — Pause auto-drive
  'r' — Resume auto-drive
  'q' — Quit

Usage:
  python tools/test_drive_detect_send.py                       # full auto
  python tools/test_drive_detect_send.py --dry-run             # no hardware
  python tools/test_drive_detect_send.py --no-depth            # skip depth, Y=0
  python tools/test_drive_detect_send.py --speed 100           # slower driving
  python tools/test_drive_detect_send.py --send-interval 3.0   # slower coord re-sends
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Camera helpers ────────────────────────────────────────────────────────────

def open_pi_cameras(cfg):
    """Open stereo Pi Camera V3 pair with ISP convergence."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("  ✗ picamera2 not installed. Run on RPi5.")
        sys.exit(1)

    W = cfg["cameras"]["stereo_left"]["width"]
    H = cfg["cameras"]["stereo_left"]["height"]

    def _open(idx, role):
        print(f"  Opening {role} camera (index {idx})...")
        cam = Picamera2(idx)
        config = cam.create_preview_configuration(
            main={"size": (2304, 1296), "format": "RGB888"}
        )
        config["controls"] = {"FrameRate": 20.0}
        cam.configure(config)
        cam.start()

        props = cam.camera_properties
        full_size = props.get("PixelArraySize", None)
        if full_size is not None and len(full_size) == 2:
            cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})

        time.sleep(2.0)
        for _ in range(15):
            cam.capture_array()
        print(f"  ✓ {role} camera ready")
        return cam

    L = _open(cfg["cameras"]["stereo_left"]["index"], "LEFT")
    R = _open(cfg["cameras"]["stereo_right"]["index"], "RIGHT")
    return L, R, W, H


def read_stereo(cam_l, cam_r, out_w, out_h):
    rgb_l = cam_l.capture_array()
    rgb_r = cam_r.capture_array()
    # Keep as RGB (no conversion to BGR), exactly how preview.py handles it
    return cv2.resize(rgb_l, (out_w, out_h)), cv2.resize(rgb_r, (out_w, out_h))







def detect_purple(frame, cfg):
    """Detect largest purple object. Returns (cx, cy, area, mask) or (None, None, 0, mask)."""
    min_area = 500
    
    # Hardcoded stronger purple thresholds (bypassing config)
    lower = np.array([130, 80, 80])
    upper = np.array([165, 255, 255])
    
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, 0, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None, None, 0, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, 0, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, int(area), mask


# ── UART char-by-char sender (proven protocol from test_detect_and_send.py) ──

def send_char_by_char(ser, x, y, z, delay_s):
    """Send coordinates char-by-char to STM32."""
    command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"
    print(f"  TX: ", end="", flush=True)

    t0 = time.perf_counter()
    for char in command_string:
        ser.write(char.encode('utf-8'))
        print(char if char != '\n' else '↵', end="", flush=True)
        time.sleep(delay_s)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  ({elapsed:.0f}ms)")
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-drive + purple detect → stop → depth → send coords to STM32")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate both ESP32 and STM32 UART")
    parser.add_argument("--no-depth", action="store_true",
                        help="Skip stereo depth, send Y=0")
    parser.add_argument("--speed", type=int, default=50,
                        help="ESP32 driving speed 0-255 (default: 50)")
    parser.add_argument("--delay", type=int, default=None,
                        help="STM32 per-char delay in ms (default: from config)")
    parser.add_argument("--cam-height", type=float, default=None,
                        help="Camera height from base in mm (default: from config)")
    parser.add_argument("--send-interval", type=float, default=2.0,
                        help="Min seconds between auto-sends while stopped (default: 2.0)")
    args = parser.parse_args()

    cfg = load_cfg()
    stm_cfg = cfg.get("stm32", {})

    stm_port = stm_cfg.get("serial_port", "/dev/ttyAMA2")
    stm_baud = stm_cfg.get("serial_baud", 115200)
    delay_ms = args.delay if args.delay is not None else stm_cfg.get("char_delay_ms", 50)
    delay_s  = delay_ms / 1000.0
    dry_run  = args.dry_run or stm_cfg.get("dry_run", False)
    cam_h    = args.cam_height if args.cam_height is not None else stm_cfg.get("cam_height_from_ref_mm", 390.0)

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  AUTO-DRIVE + PURPLE DETECT → STOP → DEPTH → STM32 SEND")
    print("=" * 64)

    # ── Connect ESP32 ─────────────────────────────────────────────────────────
    if dry_run:
        cfg.setdefault("esp32", {})["dry_run"] = True

    from comms.esp32_comms import ESP32Comms
    esp = ESP32Comms(cfg)

    print(f"  ESP32 port     : {esp.port}")
    print(f"  Drive speed    : {args.speed}")

    if not esp.connect():
        print("  ✗ ESP32 connection failed.")
        sys.exit(1)
    print("  ✓ ESP32 Connected.")
    esp.set_mode("manual")

    # ── Connect STM32 UART ────────────────────────────────────────────────────
    if dry_run:
        class FakeSerial:
            def write(self, data): pass
            def close(self): pass
        stm32 = FakeSerial()
        print(f"  ✓ STM32 UART   : DRY-RUN mode")
    else:
        try:
            import serial
            stm32 = serial.Serial(port=stm_port, baudrate=stm_baud, timeout=1)
            print(f"  ✓ STM32 UART   : {stm_port}")
        except Exception as e:
            print(f"  ✗ STM32 UART failed: {e}")
            print("    Try --dry-run to test without hardware")
            esp.close()
            sys.exit(1)

    print(f"  STM32 char delay: {delay_ms}ms")
    print(f"  Camera height  : {cam_h:.0f}mm")
    print(f"  Send interval  : {args.send_interval}s")
    print(f"  Depth          : {'DISABLED' if args.no_depth else 'ENABLED (stereo)'}")
    print(f"  Dry run        : {dry_run}")

    print("  Warming up UART (2s)...")
    time.sleep(2.0 if not dry_run else 0.1)

    # ── Open cameras ──────────────────────────────────────────────────────────
    print("\n  Opening cameras...")
    cam_l, cam_r, W, H = open_pi_cameras(cfg)

    # ── Depth engine ──────────────────────────────────────────────────────────
    depth_engine = None
    if not args.no_depth:
        try:
            from depth.stereo_depth import StereoDepth
            depth_engine = StereoDepth(cfg)
            print(f"  ✓ Depth engine loaded (f={depth_engine.focal_px:.0f}px, "
                  f"B={depth_engine.baseline_mm:.0f}mm)")
        except Exception as e:
            print(f"  ⚠ Depth engine failed: {e} — sending Y=0")

    # ── Ready ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  FLOW:")
    print("    No purple  → ESP32 drives FORWARD")
    print("    Purple seen → ESP32 STOPS → compute depth → send X#Y#Z to STM32")
    print("=" * 64)
    print("  Controls:")
    print("  's' — Pause auto-drive")
    print("  'r' — Resume auto-drive")
    print("  'q' — Quit")
    print("=" * 64 + "\n")

    # State
    is_stopped   = True
    auto_paused  = False
    send_count   = 0
    last_send_time = 0

    cv2.namedWindow("Drive+Detect+Send | s=pause r=resume q=quit", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. Capture
            left, right = read_stereo(cam_l, cam_r, W, H)
            # No software WB correction, using raw frame
            corrected = left

            # 2. Detect purple using config thresholds
            cx, cy, area, mask = detect_purple(corrected, cfg)

            vis = corrected.copy()
            purple_detected = cx is not None

            if purple_detected:
                # ── STEP A: STOP the chassis immediately ──────────────────
                if not is_stopped:
                    print(f"\n  --> 🛑 Purple @ ({cx},{cy}) area={area}px — STOPPING chassis!")
                    resp = esp.stop()
                    if resp.ok:
                        print(f"  ✓ ESP32 STOP confirmed  ({resp.latency_ms:.1f}ms)")
                    else:
                        print(f"  ✗ ESP32 STOP failed: {resp.error}")
                    is_stopped = True

                # ── STEP B: Compute depth ─────────────────────────────────
                depth_mm = 0.0
                depth_diag = ""
                if depth_engine is not None:
                    try:
                        dr = depth_engine.compute(left, right, (cx, cy), state="TEST")
                        if dr.depth_mm is not None:
                            depth_mm = dr.depth_mm
                            depth_diag = f"{depth_mm:.0f}mm (cal={dr.calibrated})"
                        elif dr.disparity_map is not None:
                            dmap = dr.disparity_map
                            win = 21
                            half = win // 2
                            y1 = max(0, cy - half)
                            y2 = min(dmap.shape[0], cy + half + 1)
                            x1 = max(0, cx - half)
                            x2 = min(dmap.shape[1], cx + half + 1)
                            patch = dmap[y1:y2, x1:x2]
                            valid = patch[patch > 0]
                            if len(valid) > 0:
                                avg_disp = float(np.median(valid))
                                depth_mm = (depth_engine.focal_px * depth_engine.baseline_mm) / avg_disp
                                depth_diag = f"{depth_mm:.0f}mm (window median)"
                            else:
                                depth_diag = f"ZERO disp @ ({cx},{cy})"
                        else:
                            depth_diag = "No disparity map"
                    except Exception as e:
                        depth_diag = f"Error: {e}"

                # ── STEP C: Compute STM32 coordinates ─────────────────────
                send_x = 90.0 - ((cx / W) * 180.0)
                send_y = depth_mm

                # ── Z: Height in arm coordinate system ─────────────────
                # Reference (0) is 390mm below camera centre.
                # Arm travels 0 → 325mm. Cannot go below 0.
                # Pinhole model: vertical offset = depth * (H/2 - cy) / focal
                CAM_HEIGHT_FROM_REF = 390.0
                ARM_MAX_MM = 325.0
                focal = depth_engine.focal_px if depth_engine else (H * 0.8)
                vert_offset = depth_mm * (H / 2.0 - cy) / focal if depth_mm > 0 else 0.0
                send_z = CAM_HEIGHT_FROM_REF + vert_offset - 115.0
                send_z = max(0.0, min(send_z, ARM_MAX_MM))

                # ── STEP D: Send coordinates to STM32 (auto, with interval) ─
                now = time.time()
                if (now - last_send_time) >= args.send_interval:
                    send_count += 1
                    print(f"\n  [#{send_count}] SEND  pixel=({cx},{cy})  "
                          f"depth={depth_mm:.0f}mm  → X={send_x:.1f} Y={send_y:.1f} Z={send_z:.1f}")
                    send_char_by_char(stm32, send_x, send_y, send_z, delay_s)
                    last_send_time = now

                # ── Display overlay ───────────────────────────────────────
                cv2.circle(vis, (cx, cy), 12, (0, 255, 0), -1)
                cv2.circle(vis, (cx, cy), 30, (255, 0, 255), 3)

                info = [
                    f"PURPLE @ ({cx},{cy})  area={area}px",
                    f"Depth: {depth_diag}" if depth_diag else "Depth: disabled",
                    f"STM32: X={send_x:.1f}  Y={send_y:.1f}  Z={send_z:.1f}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(vis, line, (10, 30 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.putText(vis, "[ STOPPED — coords sent to STM32 ]", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                # ── No purple → auto-drive forward ────────────────────────
                if not auto_paused:
                    if is_stopped:
                        print("  --> ⬆️  No purple — auto-driving FORWARD")
                        resp = esp.move("forward", args.speed)
                        if resp.ok:
                            print(f"  ✓ Moving forward ({resp.latency_ms:.1f}ms)")
                        else:
                            print(f"  ✗ Move failed: {resp.error}")
                        is_stopped = False

                    cv2.putText(vis, "No purple — DRIVING FORWARD", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                    cv2.putText(vis, "[ AUTO-DRIVE ]", (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No purple — PAUSED (press r)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(vis, "[ PAUSED ]", (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            # Send counter
            cv2.putText(vis, f"Sends: {send_count}", (W - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Drive+Detect+Send | s=pause r=resume q=quit", vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n  Quitting...")
                break
            elif key == ord('s'):
                if not is_stopped:
                    print("  --> 🛑 Manual Pause")
                    esp.stop()
                    is_stopped = True
                auto_paused = True
                print("  Auto-drive PAUSED. Press 'r' to resume.")
            elif key == ord('r'):
                auto_paused = False
                print("  Auto-drive RESUMED.")

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")

    finally:
        print("\n  Shutting down safely...")
        esp.stop()
        esp.close()
        stm32.close()
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cv2.destroyAllWindows()
        print(f"\n  Total coordinates sent: {send_count}")
        print("  ✓ All closed.\n")


if __name__ == "__main__":
    main()
