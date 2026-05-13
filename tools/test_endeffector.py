#!/usr/bin/env python3
"""
tools/test_endeffector.py — End-Effector Servo Test
===================================================
Uses the USB segmentation camera and YOLOv8n-seg model to find the main grape stem,
calculates the vertical cutting point, and moves the ESP32 end-effector servo to align.

Usage:
  python tools/test_endeffector.py                 # Live tracking mode
  python tools/test_endeffector.py --angle 22      # Manually set servo to horizontal and exit
  python tools/test_endeffector.py --angle 30      # Manually set servo angle (0-44) and exit
"""

import sys
import time
import json
import argparse
from pathlib import Path
import math

import numpy as np

import cv2
import yaml
import serial

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from segmentation.segmentor import Segmentor
from segmentation.cutting_point import extract_cutting_points

def load_cfg():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def map_value(y, in_min, in_max, out_min, out_max):
    return int((y - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def main():
    parser = argparse.ArgumentParser(description="End-Effector Servo Test")
    parser.add_argument("--angle", type=int, default=None, 
                        help="Manually send a specific angle (0-44) to the ESP32 and exit.")
    args = parser.parse_args()

    cfg = load_cfg()
    
    # ── 1. Setup Serial ───────────────────────────────────────────────────
    port = cfg["esp32"]["serial_port"]
    baud = cfg["esp32"]["serial_baud"]
    print(f"Connecting to ESP32 on {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for ESP32 to boot
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        sys.exit(1)
        
    # If manual angle provided, just send it and exit
    if args.angle is not None:
        angle = max(0, min(44, args.angle))
        data = {"cmd": "servo", "angle": angle}
        msg = json.dumps(data) + "\n"
        ser.write(msg.encode())
        print(f"Manually sent angle: {angle} to ESP32.")
        time.sleep(0.5)
        ser.close()
        sys.exit(0)

    # ── 2. Setup Camera ───────────────────────────────────────────────────
    idx = cfg["cameras"]["usb_seg"]["index"]
    w   = cfg["cameras"]["usb_seg"].get("width", 1280)
    h   = cfg["cameras"]["usb_seg"].get("height", 720)

    # Resolve symlink if needed (e.g. /dev/v4l/by-id/...)
    if isinstance(idx, str) and idx.startswith('/dev/'):
        import os
        real_idx = os.path.realpath(idx)
        if real_idx.startswith('/dev/video'):
            try:
                idx = int(real_idx.replace('/dev/video', ''))
            except ValueError:
                pass

    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print(f"Failed to open USB camera at index {idx}")
        sys.exit(1)

    print(f"USB camera opened (index={idx}, {w}x{h})")
    
    # ── 3. Setup Model ────────────────────────────────────────────────────
    print("Loading segmentation model...")
    seg = Segmentor(cfg)
    seg_cfg = cfg["segmentation"]
    cut_method = seg_cfg.get("cutting_point_method", "topmost")
    min_stem_area = seg_cfg.get("min_stem_area_px", 50)
    
    if seg._model is None:
        print("Model failed to load.")
        sys.exit(1)

    # ── 4. Load Calibration ───────────────────────────────────────────────
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
            print(f"Loaded USB calibration: fy={focal_y:.1f}, cy={principal_y:.1f}")
        except Exception as e:
            print(f"Failed to load calibration file: {e}")
    else:
        print("No calibration file found. Using linear pixel mapping fallback.")

    print("\nStarting live tracking. Press 'q' or ESC to quit.")
    
    last_send_time = 0
    current_angle = 22.0  # Start at horizontal (22 degrees)
    last_sent_angle = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        result = seg.segment(frame, state="TEST")
        
        if result.success and result.masks is not None:
            cutting = extract_cutting_points(
                result.masks,
                method=cut_method,
                min_stem_area_px=min_stem_area,
            )
            
            if cutting and cutting.primary:
                # We have a valid primary cutting point
                cy = cutting.primary.py
                cx = cutting.primary.px
                
                # ── Visual Servoing (Proportional Control) ──
                # 22 degrees = horizontal (blue line). Range: 0–44.
                # Target is the blue line (principal_y if calibrated, else h//2)
                target_y = principal_y if calibrated else h // 2
                
                # Calculate error in pixels.
                # Positive error = cutting point is BELOW blue line (servo needs to tilt down).
                error_y = cy - target_y
                deadzone = 15  # deadzone in pixels to prevent jitter
                
                if abs(error_y) > deadzone:
                    kp = 0.03  # Proportional gain
                    delta = error_y * kp
                    
                    # Constrain the increment speed between 1 and 3 degrees to move slowly
                    if delta > 0:
                        delta = min(3.0, max(1.0, delta))
                    else:
                        delta = max(-3.0, min(-1.0, delta))
                        
                    current_angle += delta
                    current_angle = max(0.0, min(44.0, current_angle))
                
                angle = int(current_angle)

                # Throttle serial messages (e.g. max 10 Hz) and only send if angle changes
                if time.time() - last_send_time > 0.1 and abs(angle - last_sent_angle) >= 1:
                    data = {
                        "cmd": "servo",
                        "angle": angle
                    }
                    try:
                        msg = json.dumps(data) + "\n"
                        ser.write(msg.encode())
                        print(f"Sent Angle: {angle} (error={error_y:.1f}px)")
                        last_send_time = time.time()
                        last_sent_angle = angle
                    except Exception as e:
                        print(f"Serial write error: {e}")

                # Draw debug
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(frame, (0, cy), (w, cy), (0, 0, 255), 1)
                cv2.putText(frame, f"Angle: {angle} | err: {error_y:.0f}px", (cx + 15, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw masks (outside cutting.primary so they always show)
            for m in result.masks:
                mask_overlay = frame.copy()
                mask_overlay[m] = (0, 255, 0)
                cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)

        # Draw center reference (blue line = 22 degrees / horizontal)
        cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 2)
        cv2.putText(frame, "22 deg (ref)", (10, h//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Read incoming serial to keep buffer clear (suppressed printing to reduce terminal spam)
        while ser.in_waiting > 0:
            try:
                ser.readline()
            except Exception:
                pass

        # HUD
        cv2.putText(frame, f"Servo: {int(current_angle)} deg", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "q/ESC: quit | r: reset", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("End-Effector Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # ESC or 'q'
            break
        elif key == ord('r'):
            current_angle = 22.0
            ser.write(json.dumps({"cmd": "servo", "angle": 22}).encode() + b"\n")
            print("Reset servo to 22° (Horizontal)")

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
