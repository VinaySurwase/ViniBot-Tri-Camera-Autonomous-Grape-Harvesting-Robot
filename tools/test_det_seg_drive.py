#!/usr/bin/env python3
"""
tools/test_grape_stop.py — Full Autonomous Pipeline Test
========================================================
End-to-End ViniBot Workflow:
  1. SEARCH: Drives forward using ESP32. YOLO detects grape cluster.
  2. STOP & DEPTH: Stops ESP32. Uses Stereo Depth to find target coordinates.
  3. ARM MOVE: Sends coordinates to STM32 and waits for arm to reach.
  4. VISUAL SERVO: Uses USB segmentation to track cutting point and adjusts end-effector.

Usage:
  python tools/test_grape_stop.py
"""

import argparse
import sys
import time
import json
from pathlib import Path
import math

import cv2
import numpy as np
import yaml
import serial

sys.path.insert(0, str(Path(__file__).parent.parent))

from comms.esp32_comms import ESP32Comms
from depth.stereo_depth import StereoDepth
from segmentation.segmentor import Segmentor
from segmentation.cutting_point import extract_cutting_points

# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def open_pi_cameras(cfg):
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
    return cv2.resize(rgb_l, (out_w, out_h)), cv2.resize(rgb_r, (out_w, out_h))

def send_char_by_char(ser, x, y, z, delay_s):
    command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"
    print(f"  STM32 TX: ", end="", flush=True)
    t0 = time.perf_counter()
    for char in command_string:
        ser.write(char.encode('utf-8'))
        print(char if char != '\n' else '↵', end="", flush=True)
        time.sleep(delay_s)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  ({elapsed:.0f}ms)")
    return elapsed

# ── Custom Target Detector ────────────────────────────────────────────────────

class TargetDecision:
    def __init__(self, found=False, box=None, center=None):
        self.cluster_found = found
        self.cluster_box = box
        self.cluster_centre = center
        self.confirmed = found

class TargetDetector:
    def __init__(self):
        self.confirm_count = 0
        # CLAHE for normalizing brightness across lighting conditions
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def process(self, frame, state="TEST"):
        # 1. Normalize brightness so detection works in daylight & artificial light
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Convert to HSV
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

        # 3. Dual-range mask for deep royal blue
        #    Range 1: Core deep blue   (H 100-130)
        #    Range 2: Warm-light blue  (H 130-150)
        #    Saturation ≥ 80 to reject skin, greys, desaturated background
        #    Value ≥ 80 to reject black objects even after CLAHE normalization
        mask1 = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([130, 80, 80]), np.array([150, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 4. Clean up — larger kernel to reject noise, keep solid blobs
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 3000:  # min area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w // 2
                cy = y + h // 2
                
                self.confirm_count += 1
                decision = TargetDecision(found=True, box=(x, y, x+w, y+h), center=(cx, cy))
                decision.confirmed = (self.confirm_count >= 3)
                return decision
                
        self.confirm_count = 0
        return TargetDecision(found=False)
        
    def reset_confirmation(self):
        self.confirm_count = 0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full Autonomous Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Simulate hardware")
    parser.add_argument("--speed", type=int, default=50, help="Driving speed")
    parser.add_argument("--arm-wait", type=int, default=20, help="Seconds to wait for arm")
    args = parser.parse_args()

    cfg = load_cfg()

    print("\n" + "=" * 60)
    print("  VINIBOT FULL PIPELINE")
    print("=" * 60)

    # 1. Connect ESP32
    esp = ESP32Comms(cfg)
    if not esp.connect() and not args.dry_run:
        print("  ✗ Failed to connect to ESP32.")
        sys.exit(1)
    esp.set_mode("manual")
    # Reset servo to horizontal
    esp._send({"cmd": "servo", "angle": 22})

    # 2. Connect STM32
    stm_cfg = cfg.get("stm32", {})
    port_stm = stm_cfg.get("serial_port", "/dev/ttyAMA2")
    baud_stm = stm_cfg.get("serial_baud", 115200)
    delay_ms_stm = stm_cfg.get("char_delay_ms", 50)
    delay_s_stm = delay_ms_stm / 1000.0
    
    if args.dry_run:
        class FakeSerial:
            def write(self, data): pass
            def close(self): pass
        stm32 = FakeSerial()
    else:
        try:
            stm32 = serial.Serial(port=port_stm, baudrate=baud_stm, timeout=1)
        except Exception as e:
            print(f"  ✗ Failed to connect to STM32: {e}")
            sys.exit(1)

    # 3. Load Models & Engines
    print("  Loading AI Models...")
    det = TargetDetector()
    seg = Segmentor(cfg)
    try:
        depth_engine = StereoDepth(cfg)
    except Exception as e:
        print(f"  ⚠ Depth engine error: {e}")
        depth_engine = None

    # 4. Open Cameras
    print("  Opening cameras...")
    cam_l, cam_r, W, H = open_pi_cameras(cfg)

    idx_usb = cfg["cameras"]["usb_seg"]["index"]
    if isinstance(idx_usb, str) and idx_usb.startswith("/dev/"):
        import os
        real_idx = os.path.realpath(idx_usb)
        if real_idx.startswith("/dev/video"):
            try: idx_usb = int(real_idx.replace("/dev/video", ""))
            except: pass
    cap_usb = cv2.VideoCapture(idx_usb)
    w_usb, h_usb = 1280, 720
    cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, w_usb)
    cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, h_usb)

    # 5. USB Camera Calibration
    calib_file = Path(__file__).parent.parent / "calibration" / "usb_calib.npz"
    calibrated_usb = False
    if calib_file.exists():
        data = np.load(calib_file)
        principal_y = float(data["cy"])
        calibrated_usb = True
    else:
        principal_y = h_usb // 2

    # ── State Machine ──
    STATE_SEARCHING = 0
    STATE_ARM_MOVING = 1
    STATE_SERVOING = 2
    STATE_CUTTING = 3
    STATE_PAUSED = 99

    state = STATE_SEARCHING
    saved_state = STATE_SEARCHING   # remembers state before pause
    is_stopped = True
    arm_wait_start = 0

    current_servo_angle = 22.0
    last_servo_send = 0
    last_sent_angle = -1
    servo_aligned_time = 0        # tracks when servo first entered deadzone
    cut_count = 0

    cv2.namedWindow("ViniBot Pipeline", cv2.WINDOW_NORMAL)
    print("\n  >>> SYSTEM READY. Commencing pipeline. <<<")

    try:
        while True:
            # ──────────────────────────────────────────────────────────────────
            if state == STATE_SEARCHING:
                left, right = read_stereo(cam_l, cam_r, W, H)
                vis = left.copy()
                decision = det.process(left, state="TEST")

                if decision.cluster_found:
                    # 🛑 1. STOP CHASSIS
                    if not is_stopped:
                        print("  🛑 Target detected! Stopping chassis...")
                        esp.stop()
                        is_stopped = True

                    x1, y1, x2, y2 = decision.cluster_box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cx, cy = decision.cluster_centre
                    cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)

                    if decision.confirmed:
                        # 📏 2. COMPUTE DEPTH
                        depth_mm = 0.0
                        if depth_engine:
                            dr = depth_engine.compute(left, right, (cx, cy), state="TEST")
                            if dr.depth_mm is not None:
                                depth_mm = dr.depth_mm
                            elif dr.disparity_map is not None:
                                dmap = dr.disparity_map
                                patch = dmap[max(0, cy-10):min(H, cy+11), max(0, cx-10):min(W, cx+11)]
                                valid = patch[patch > 0]
                                if len(valid) > 0:
                                    avg_disp = float(np.median(valid))
                                    depth_mm = (depth_engine.focal_px * depth_engine.baseline_mm) / avg_disp

                        # 🤖 3. SEND TO STM32
                        if 150 < depth_mm <= 650:
                            send_x = 90.0 - ((cx / W) * 180.0)
                            send_y = depth_mm

                            # ── Z: Height in arm coordinate system ────────
                            # Reference (0) is 390mm below camera centre.
                            # Arm travels 0 → 325mm. Cannot go below 0.
                            # Pinhole model: vertical offset = depth * (H/2 - cy) / focal
                            #   cy < H/2 → target above camera centre → higher Z
                            #   cy > H/2 → target below camera centre → lower Z
                            CAM_HEIGHT_FROM_REF = 390.0
                            ARM_MAX_MM = 325.0
                            focal = depth_engine.focal_px if depth_engine else (H * 0.8)
                            vert_offset = depth_mm * (H / 2.0 - cy) / focal
                            send_z = CAM_HEIGHT_FROM_REF + vert_offset - 115.0
                            send_z = max(0.0, min(send_z, ARM_MAX_MM))

                            print(f"  --> Arm Target: X={send_x:.1f} Y={send_y:.1f} Z={send_z:.1f}")
                            send_char_by_char(stm32, send_x, send_y, send_z, delay_s_stm)

                            state = STATE_ARM_MOVING
                            arm_wait_start = time.time()
                            print(f"  --> Waiting {args.arm_wait}s for arm to move...")
                        else:
                            cv2.putText(vis, f"Waiting for depth (Current: {depth_mm:.1f}mm)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                else:
                    # ⬆️ DRIVE FORWARD
                    if is_stopped:
                        print("  ⬆️ No target found. Driving forward...")
                        esp.move("forward", args.speed)
                        is_stopped = False
                        det.reset_confirmation()
                    cv2.putText(vis, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("ViniBot Pipeline", vis)

            # ──────────────────────────────────────────────────────────────────
            elif state == STATE_ARM_MOVING:
                # Read frames to keep buffers clear
                read_stereo(cam_l, cam_r, W, H)
                cap_usb.read()

                vis = np.zeros((H, W, 3), dtype=np.uint8)
                remain = args.arm_wait - (time.time() - arm_wait_start)
                cv2.putText(vis, f"ARM MOVING... {remain:.1f}s", (50, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("ViniBot Pipeline", vis)

                if remain <= 0:
                    print("  --> Arm move complete. Switching to visual servoing.")
                    state = STATE_SERVOING

            # ──────────────────────────────────────────────────────────────────
            elif state == STATE_SERVOING:
                ret, frame = cap_usb.read()
                if not ret: continue

                vis = frame.copy()
                result = seg.segment(frame, state="TEST")

                if result.success and result.masks is not None:
                    cutting = extract_cutting_points(result.masks, method="topmost", min_stem_area_px=50)

                    if cutting and cutting.primary:
                        cy = cutting.primary.py
                        cx = cutting.primary.px

                        target_y = principal_y if calibrated_usb else h_usb // 2
                        error_y = cy - target_y

                        if abs(error_y) > 15:  # deadzone
                            delta = error_y * 0.03
                            if delta > 0: delta = min(3.0, max(1.0, delta))
                            else:         delta = max(-3.0, min(-1.0, delta))
                            current_servo_angle += delta
                            current_servo_angle = max(0.0, min(44.0, current_servo_angle))

                        angle = int(current_servo_angle)

                        if time.time() - last_servo_send > 5.0 and abs(angle - last_sent_angle) >= 1:
                            if angle > last_sent_angle:
                                step_angle = last_sent_angle + 1
                            else:
                                step_angle = last_sent_angle - 1
                            step_angle = max(0, min(44, step_angle))

                            esp._send({"cmd": "servo", "angle": step_angle})
                            print(f"  [SERVO] Step: {last_sent_angle}° → {step_angle}° (target={angle}°, err={error_y:.0f}px)")
                            last_servo_send = time.time()
                            last_sent_angle = step_angle

                            # Start alignment timer on first servo send
                            if servo_aligned_time == 0:
                                servo_aligned_time = time.time()

                        cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
                        cv2.putText(vis, f"Servo: {angle}°", (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    for m in result.masks:
                        overlay = vis.copy()
                        overlay[m] = (0, 255, 0)
                        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

                # After first servo send, wait 3 seconds then cut
                if servo_aligned_time > 0 and time.time() - servo_aligned_time > 3.0:
                    print("  ✂  Servo adjusted. Starting cut sequence...")
                    state = STATE_CUTTING
                    cut_count = 0
                    continue

                target_y = int(principal_y if calibrated_usb else h_usb // 2)
                cv2.line(vis, (0, target_y), (w_usb, target_y), (255, 0, 0), 2)
                remain = max(0, 3.0 - (time.time() - servo_aligned_time)) if servo_aligned_time > 0 else 0
                label = f"VISUAL SERVOING — Cut in {remain:.1f}s" if servo_aligned_time > 0 else "VISUAL SERVOING"
                cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("ViniBot Pipeline", vis)

            # ──────────────────────────────────────────────────────────────────
            elif state == STATE_CUTTING:
                cut_count += 1
                print(f"  ✂  CUT {cut_count}/3")
                resp = esp._send({"cmd": "cut"})
                if resp.ok:
                    print(f"     ✓ Cut {cut_count} done ({resp.latency_ms:.0f}ms)")
                else:
                    print(f"     ✗ Cut {cut_count} failed: {resp.error}")

                if cut_count >= 3:
                    print("  ✓ Cut sequence complete. Returning to search...")
                    # Reset servo to center
                    esp._send({"cmd": "servo", "angle": 22})
                    current_servo_angle = 22.0
                    last_sent_angle = -1
                    servo_aligned_time = 0
                    is_stopped = True
                    det.reset_confirmation()
                    state = STATE_SEARCHING
                else:
                    time.sleep(1.0)  # pause between cuts

            # ──────────────────────────────────────────────────────────────────
            elif state == STATE_PAUSED:
                vis = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(vis, "PAUSED - Press [w] to resume", (50, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("ViniBot Pipeline", vis)
                # Drain camera buffers so they don't stale
                try:
                    read_stereo(cam_l, cam_r, W, H)
                    cap_usb.read()
                except: pass

            # ──────────────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s') and state != STATE_PAUSED:
                esp.stop()
                is_stopped = True
                saved_state = state
                state = STATE_PAUSED
                print("  🛑 PAUSED — wheels stopped. Press [w] to resume.")
            elif key == ord('w') and state == STATE_PAUSED:
                state = saved_state
                is_stopped = True   # let the pipeline re-send forward when needed
                print("  ▶  RESUMED")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("\n  Shutting down...")
        esp.stop()
        esp.close()
        stm32.close()
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cap_usb.release()
        cv2.destroyAllWindows()
        print("  ✓ System closed safely.")

if __name__ == "__main__":
    main()
