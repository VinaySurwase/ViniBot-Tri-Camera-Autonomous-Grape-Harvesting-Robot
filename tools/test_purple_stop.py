#!/usr/bin/env python3
"""
tools/test_purple_stop.py
=========================
Combines PiCamera2 purple detection with the ESP32 chassis controller.
Drives the robot forward and automatically sends STOP to ESP32 the
moment a purple object is detected in camera feed.

Camera config and purple detection are identical to test_detect_and_send.py.

Controls (while the OpenCV window is focused):
  'f' — Drive Forward
  'b' — Drive Backward
  's' — Manual Stop
  'q' — Quit

Usage:
  python tools/test_purple_stop.py
  python tools/test_purple_stop.py --dry-run   # simulate without ESP32 hardware
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


# ── Camera helpers — IDENTICAL to test_detect_and_send.py ────────────────────

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

        # Force full sensor crop so we don't lose wide FOV due to implicit crop.
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





# ── Purple detection — IDENTICAL to test_detect_and_send.py ──────────────────


def detect_purple(frame, cfg):
    """
    Detect largest purple object using HSV thresholding.
    Uses config.yaml HSV range.
    Returns (cx, cy, area, mask) or (None, None, 0, mask).
    """
    min_area = 500   # same as test_detect_and_send.py

    # Hardcoded stronger purple thresholds (bypassing config)
    lower = np.array([130, 80, 80])
    upper = np.array([165, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Only MORPH_OPEN — same as test_detect_and_send.py (no CLOSE)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect purple → send STOP to ESP32")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without ESP32 hardware")
    parser.add_argument("--speed", type=int, default=50,
                        help="Driving speed 0-255 (default: 50)")
    args = parser.parse_args()

    cfg = load_cfg()

    # ── Connect ESP32 ─────────────────────────────────────────────────────────
    if args.dry_run:
        cfg.setdefault("esp32", {})["dry_run"] = True

    from comms.esp32_comms import ESP32Comms
    esp = ESP32Comms(cfg)

    print("\n" + "=" * 55)
    print("  Purple Auto-Stop Tester")
    print("=" * 55)
    print(f"  ESP32 port : {esp.port}")
    print(f"  Dry run    : {esp.dry_run}")
    print("=" * 55)

    if not esp.connect():
        print("  ✗ Failed to connect to ESP32. Check USB connection and config.yaml.")
        sys.exit(1)
    print("  ✓ ESP32 Connected.")

    # Start in MANUAL mode so the Pi has full control
    esp.set_mode("manual")
    print("  ✓ ESP32 set to MANUAL mode.\n")

    # ── Open cameras ──────────────────────────────────────────────────────────
    print("  Opening cameras...")
    cam_l, cam_r, W, H = open_pi_cameras(cfg)

    print("\n" + "=" * 55)
    print("  Purple Auto-Stop Tester")
    print("  Bot drives forward automatically.")
    print("  Stops the moment purple is detected.")
    print("=" * 55)
    print("  Controls (click OpenCV window first!):")
    print("  's' — Manual Stop / Pause auto-drive")
    print("  'r' — Resume auto-drive")
    print("  'q' — Quit")
    print("=" * 55 + "\n")

    # Track drive state to avoid spamming commands every loop tick
    is_stopped = True
    # Set to False if user manually pauses auto-drive with 's'
    auto_paused = False
    current_speed = args.speed

    cv2.namedWindow("Purple Auto-Stop | s=pause r=resume q=quit", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. Grab frames from both cameras (same as test_detect_and_send.py)
            left, _right = read_stereo(cam_l, cam_r, W, H)

            # No software WB correction, using raw frame
            corrected = left

            # 3. Detect purple on the frame using config
            cx, cy, area, mask = detect_purple(corrected, cfg)

            # 4. Show corrected frame in preview (not the raw blue one)
            vis = corrected.copy()
            purple_detected = cx is not None
            
            # Display current speed
            cv2.putText(vis, f"Speed: {current_speed}", (W - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if purple_detected:
                # Draw detection overlay
                cv2.circle(vis, (cx, cy), 12, (0, 255, 0), -1)
                cv2.circle(vis, (cx, cy), 30, (255, 0, 255), 3)

                cv2.putText(vis, f"PURPLE DETECTED  area={area}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(vis, f"Center: ({cx}, {cy})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # If robot is currently driving → STOP immediately
                if not is_stopped:
                    print(f"  --> 🛑 Purple @ ({cx},{cy}) area={area}px — Sending STOP to ESP32!")
                    resp = esp.stop()
                    if resp.ok:
                        print(f"  ✓ ESP32 confirmed STOP  ({resp.latency_ms:.1f}ms)")
                    else:
                        print(f"  ✗ ESP32 STOP failed: {resp.error}")
                    is_stopped = True

                cv2.putText(vis, "[ AUTO-STOPPED — purple seen ]" , (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            else:
                # No purple detected — drive forward automatically unless paused
                if not auto_paused:
                    if is_stopped:
                        # Only send the command on the transition stopped→moving
                        print(f"  --> ⬆️  No purple — auto-driving FORWARD (speed={current_speed})")
                        resp = esp.move("forward", current_speed)
                        if resp.ok:
                            print(f"  ✓ Moving forward  ({resp.latency_ms:.1f}ms)")
                        else:
                            print(f"  ✗ Move failed: {resp.error}")
                        is_stopped = False

                    cv2.putText(vis, "No purple — DRIVING FORWARD", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                    cv2.putText(vis, "[ AUTO-DRIVE ]", (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No purple — PAUSED (press r to resume)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(vis, "[ PAUSED ]", (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            cv2.imshow("Purple Auto-Stop | s=pause r=resume q=quit", vis)
            key = cv2.waitKey(30) & 0xFF

            # Keyboard controls
            if key == ord('q'):
                print("\n  Quitting...")
                break

            elif key == ord('s'):
                # Manually pause auto-drive
                if not is_stopped:
                    print("  --> 🛑 Manual Pause")
                    esp.stop()
                    is_stopped = True
                auto_paused = True
                print("  Auto-drive PAUSED. Press 'r' to resume.")

            elif key == ord('r'):
                # Resume auto-drive
                auto_paused = False
                print("  Auto-drive RESUMED.")

            elif key == ord('+') or key == ord('='):
                current_speed = min(255, current_speed + 10)
                print(f"  Speed increased to: {current_speed}")
                if not is_stopped and not auto_paused and not purple_detected:
                    esp.move("forward", current_speed)
                    
            elif key == ord('-') or key == ord('_'):
                current_speed = max(0, current_speed - 10)
                print(f"  Speed decreased to: {current_speed}")
                if not is_stopped and not auto_paused and not purple_detected:
                    esp.move("forward", current_speed)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")

    finally:
        print("\n  Shutting down safely...")
        esp.stop()
        esp.close()
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cv2.destroyAllWindows()
        print("  ✓ All closed.\n")


if __name__ == "__main__":
    main()
