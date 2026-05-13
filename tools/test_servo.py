#!/usr/bin/env python3
"""
tools/test_servo.py - Standalone End-Effector Servo Tester (Python Side)
========================================================================
Pairs with: esp32/servo_test/servo_test.ino  (standalone)
       or:  esp32/vinibot_esp32/vinibot_esp32.ino (full firmware)

Two modes:
  1. Manual angle - Send a specific angle and see the response.
  2. Live tracking - Use the USB camera + YOLOv8n-seg to track the grape
                     stem cutting point and incrementally move the servo.

The servo reference frame (44-degree symmetric setup):
  * 0 degrees  = Max upward tilt
  * 22 degrees = Horizontal (the blue reference line on screen)
  * 44 degrees = Max downward tilt

Usage:
  python tools/test_servo.py --angle 30          # Send angle 30 and exit
  python tools/test_servo.py --angle 22          # Reset to horizontal
  python tools/test_servo.py --sweep             # Run a full 0-44-0 sweep
  python tools/test_servo.py --live              # Live camera tracking mode

  python tools/test_servo.py --live --port /dev/ttyUSB0  # Override serial port
"""

import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import serial
import yaml

def load_cfg():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def open_serial(port, baud):
    """Open serial port with boot wait."""
    print(f" Connecting to ESP32 on {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # ESP32 boot time
        # Drain any boot messages
        while ser.in_waiting:
            ser.readline()
        print(" ✓ Connected.\n")
        return ser
    except Exception as e:
        print(f" ✗ Failed to open serial port: {e}")
        sys.exit(1)

def send_cmd(ser, cmd_dict):
    """Send a JSON command and wait for the response."""
    msg = json.dumps(cmd_dict) + "\n"
    ser.write(msg.encode())
    # Wait for reply (up to 3 seconds for sweep)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if ser.in_waiting:
            try:
                line = ser.readline().decode().strip()
                if line.startswith("{"):
                    resp = json.loads(line)
                    return resp
            except Exception:
                pass
        time.sleep(0.05)
    return None

def drain_serial(ser):
    """Clear incoming buffer without printing."""
    while ser.in_waiting:
        try:
            ser.readline()
        except Exception:
            pass


# ======================================================================
# MODE 1: Manual Angle
# ======================================================================
def mode_angle(ser, angle):
    # Enforce the 44-degree safety limit for manual inputs too
    angle = max(0, min(44, angle))
    print(f" Sending angle: {angle}°")
    resp = send_cmd(ser, {"cmd": "servo", "angle": angle})
    if resp and resp.get("ok"):
        print(f" ✓ ESP32 confirmed -> {resp.get('angle', '?')}°")
    else:
        print(" ⚠ No valid response (timeout or error)")

    # Hold for 10 seconds so the servo stays at this angle for observation
    print(f" Holding at {angle}° for 10 seconds (Ctrl+C to skip)...")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    print(" Done.")

# ======================================================================
# MODE 2: Sweep Test
# ======================================================================
def mode_sweep(ser):
    print(" Starting sweep (0 -> 44 -> 0)...")
    resp = send_cmd(ser, {"cmd": "sweep"})
    if resp and resp.get("ok"):
        print(f" ✓ Sweep complete -> {resp.get('msg', '')}")
    else:
        print(" ⚠ No valid response (timeout or error)")

# ======================================================================
# MODE 3: Live Camera Tracking
# ======================================================================
def mode_live(ser, cfg):
    import cv2
    import math
    import numpy as np
    from segmentation.segmentor import Segmentor
    from segmentation.cutting_point import extract_cutting_points

    # --- Camera setup ---
    idx = cfg["cameras"]["usb_seg"]["index"]
    w = cfg["cameras"]["usb_seg"].get("width", 1280)
    h = cfg["cameras"]["usb_seg"].get("height", 720)

    if isinstance(idx, str) and idx.startswith("/dev/"):
        import os
        real_idx = os.path.realpath(idx)
        if real_idx.startswith("/dev/video"):
            try:
                idx = int(real_idx.replace("/dev/video", ""))
            except ValueError:
                pass

    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print(f" ✗ Failed to open USB camera at index {idx}")
        return

    print(f" ✓ USB camera opened ({w}×{h})")

    # --- Model setup ---
    print(" Loading segmentation model...")
    seg = Segmentor(cfg)
    seg_cfg = cfg["segmentation"]
    cut_method = seg_cfg.get("cutting_point_method", "topmost")
    min_stem_area = seg_cfg.get("min_stem_area_px", 50)

    # --- Calibration (optional) ---
    calib_file = PROJECT_ROOT / "calibration" / "usb_calib.npz"
    calibrated = False
    focal_y = None
    principal_y = None
    if calib_file.exists():
        try:
            data = np.load(calib_file)
            focal_y = float(data["fy"])
            principal_y = float(data["cy"])
            calibrated = True
            print(f" Loaded USB calibration: fy={focal_y:.1f}, cy={principal_y:.1f}")
        except Exception as e:
            print(f" Failed to load calibration: {e}")
    else:
        print(" No calibration file found. Using center of frame as reference.")

    # --- Tracking state ---
    current_angle = 22.0  # Start at horizontal (22 degrees)
    last_sent_angle = -1
    last_send_time = 0
    target_y = principal_y if calibrated else h // 2

    print("\n ┌────────────────────────────────────────────────────────┐")
    print(" │ LIVE TRACKING MODE                                     │")
    print(" │ Blue line = 22° (horizontal reference)                 │")
    print(" │ Servo moves +1 to +3 deg/frame toward target           │")
    print(" │ Press 'q' or ESC to quit                               │")
    print(" │ Press 'r' to reset servo to 22°                        │")
    print(" └────────────────────────────────────────────────────────┘\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Inference ---
        result = seg.segment(frame, state="TEST")

        if result.success and result.masks is not None:
            cutting = extract_cutting_points(
                result.masks,
                method=cut_method,
                min_stem_area_px=min_stem_area,
            )

            if cutting and cutting.primary:
                cy = cutting.primary.py
                cx = cutting.primary.px

                # --- Visual Servoing (Proportional Control) ---
                # Positive error = cutting point is BELOW blue line (servo tilts down)
                error_y = cy - target_y
                deadzone = 15  # pixels

                if abs(error_y) > deadzone:
                    kp = 0.03
                    delta = error_y * kp
                    # Clamp speed to 1-3 deg/frame for gentle movement
                    if delta > 0:
                        delta = min(3.0, max(1.0, delta))
                    else:
                        delta = max(-3.0, min(-1.0, delta))
                    current_angle += delta
                    current_angle = max(0.0, min(44.0, current_angle))

                angle = int(current_angle)

                # Throttle serial (max ~10 Hz, only if angle changed)
                now = time.time()
                if now - last_send_time > 0.1 and abs(angle - last_sent_angle) >= 1:
                    msg = json.dumps({"cmd": "servo", "angle": angle}) + "\n"
                    try:
                        ser.write(msg.encode())
                        print(f" Sent: {angle}° (error={error_y:.0f}px)")
                        last_send_time = now
                        last_sent_angle = angle
                    except Exception as e:
                        print(f" Serial write error: {e}")

                # --- Draw debug ---
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(frame, (0, cy), (w, cy), (0, 0, 255), 1)
                label = f"Angle: {angle} | err: {error_y:.0f}px"
                cv2.putText(frame, label, (cx + 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- Draw masks (outside cutting.primary so they always show) ---
            for m in result.masks:
                overlay = frame.copy()
                overlay[m] = (0, 255, 0)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # --- Blue reference line ---
        cv2.line(frame, (0, int(target_y)), (w, int(target_y)), (255, 0, 0), 2)
        cv2.putText(frame, "22 deg (ref)", (10, int(target_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # --- HUD ---
        cv2.putText(frame, f"Servo: {int(current_angle)} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "q/ESC: quit | r: reset", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Drain serial buffer
        drain_serial(ser)

        cv2.imshow("Servo Live Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("r"):
            current_angle = 22.0
            ser.write(json.dumps({"cmd": "servo", "angle": 22}).encode() + b"\n")
            print(" Reset servo to 22° (Horizontal)")

    cap.release()
    cv2.destroyAllWindows()

# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Standalone End-Effector Servo Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python tools/test_servo.py --angle 30          Send angle 30 and exit
  python tools/test_servo.py --angle 22          Reset to horizontal
  python tools/test_servo.py --sweep             Full 0-44-0 sweep
  python tools/test_servo.py --live              Live camera tracking

  python tools/test_servo.py --live --port /dev/ttyUSB0
  """
    )
    parser.add_argument("--angle", type=int, default=None,
                        help="Send a specific angle (0-44) and exit.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run a full 0-44-0 sweep to visually verify servo.")
    parser.add_argument("--live", action="store_true",
                        help="Launch live camera tracking mode.")
    parser.add_argument("--port", type=str, default=None,
                        help="Override serial port (default: from config.yaml).")
    parser.add_argument("--baud", type=int, default=None,
                        help="Override baud rate (default: from config.yaml).")
    args = parser.parse_args()

    if args.angle is None and not args.sweep and not args.live:
        parser.print_help()
        print("\n ⚠ Please specify --angle, --sweep, or --live.")
        sys.exit(1)

    cfg = load_cfg()
    port = args.port or cfg["esp32"]["serial_port"]
    baud = args.baud or cfg["esp32"]["serial_baud"]

    print()
    print(" ╔════════════════════════════════════════════════════════╗")
    print(" ║  ViniBot End-Effector Servo Tester (Python)            ║")
    print(" ╠════════════════════════════════════════════════════════╣")
    print(f" ║  Port : {port:<42}║")
    print(f" ║  Baud : {baud:<42}║")
    print(" ╚════════════════════════════════════════════════════════╝")
    print()

    ser = open_serial(port, baud)

    try:
        if args.angle is not None:
            mode_angle(ser, args.angle)

        elif args.sweep:
            mode_sweep(ser)

        elif args.live:
            mode_live(ser, cfg)

    except KeyboardInterrupt:
        print("\n Interrupted. Exiting.")
    finally:
        ser.close()
        print(" Serial port closed.\n")


if __name__ == "__main__":
    main()
