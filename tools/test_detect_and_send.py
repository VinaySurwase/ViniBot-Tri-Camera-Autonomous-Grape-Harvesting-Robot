#!/usr/bin/env python3
"""
tools/test_detect_and_send.py
==============================
End-to-end test: Detect purple grape cluster → compute depth → send
coordinates to STM32 over UART.

Flow:
  1. Open stereo Pi cameras (LEFT + RIGHT)
  2. Detect purple objects via HSV thresholding
  3. Compute depth from stereo disparity
  4. Map pixel position → STM32 coordinate system
  5. Send (X, Y, Z) to STM32 using proven char-by-char protocol

STM32 coordinate system (what gets sent as X#Y#Z):
  X = horizontal position (0.0 = centre; positive=left, negative=right)
      mapped linearly from pixel cx: X = 90 - (cx / frame_width) * 180
  Y = depth in mm from stereo engine (how far away the target is)
  Z = height in arm coordinate system (0–325mm)
      Reference (0) is 390mm below camera centre.
      Pinhole model: Z = 390 + depth * (H/2 - cy) / focal_px

Usage:
  python tools/test_detect_and_send.py                     # live detect + auto-send
  python tools/test_detect_and_send.py --manual             # SPACE to send each detection
  python tools/test_detect_and_send.py --dry-run            # simulate UART
  python tools/test_detect_and_send.py --delay 500          # match reference timing
  python tools/test_detect_and_send.py --no-depth           # skip depth, send Y=0

Press 'q' to quit, SPACE to send (in manual mode).
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
        from libcamera import controls as libcam_controls
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
        config["controls"] = {
            "FrameRate": 20.0,
        }
        cam.configure(config)
        cam.start()

        # Force full sensor crop so we don't lose wide FOV due to implicit crop.
        props = cam.camera_properties
        full_size = props.get("PixelArraySize", None)
        if full_size is not None and len(full_size) == 2:
            cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})

        time.sleep(2.0)  # ISP convergence
        for _ in range(15):
            cam.capture_array()
        print(f"  ✓ {role} camera ready")
        return cam

    L = _open(cfg["cameras"]["stereo_left"]["index"], "LEFT")
    R = _open(cfg["cameras"]["stereo_right"]["index"], "RIGHT")
    return L, R, W, H


def read_stereo(cam_l, cam_r, out_w, out_h):
    """Capture one frame from each camera, keep as RGB (like preview.py)."""
    rgb_l = cam_l.capture_array()
    rgb_r = cam_r.capture_array()
    return cv2.resize(rgb_l, (out_w, out_h)), cv2.resize(rgb_r, (out_w, out_h))





# ── Purple detection (matched to test_purple_distance.py) ─────────────────────


def detect_purple(frame, cfg):
    """
    Detect largest purple object using HSV thresholding.
    Uses config.yaml HSV range.
    Returns (cx, cy, area, mask) or (None, None, 0, mask).
    """
    min_area = 500   # same as test_purple_distance.py

    # Hardcoded stronger purple thresholds (bypassing config)
    lower = np.array([130, 80, 80])
    upper = np.array([165, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Only MORPH_OPEN — same as test_purple_distance.py (no CLOSE)
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


# ── UART char-by-char sender ─────────────────────────────────────────────────

def send_char_by_char(ser, x, y, z, delay_s):
    """Send coordinates char-by-char — proven protocol from uart_send.py."""
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
        description="Detect purple → compute depth → send to STM32")
    parser.add_argument("--manual", action="store_true",
                        help="Press SPACE to send each detection (default: auto-send)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate UART (no hardware needed)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Skip stereo depth, send Z=0")
    parser.add_argument("--delay", type=int, default=None,
                        help="Per-char delay in ms (default: from config)")
    parser.add_argument("--cam-height", type=float, default=None,
                        help="Camera height from base in mm (default: from config)")
    parser.add_argument("--send-interval", type=float, default=2.0,
                        help="Min seconds between auto-sends (default: 2.0)")
    args = parser.parse_args()

    cfg = load_cfg()
    stm_cfg = cfg.get("stm32", {})

    port     = stm_cfg.get("serial_port", "/dev/ttyAMA2")
    baud     = stm_cfg.get("serial_baud", 115200)
    delay_ms = args.delay if args.delay is not None else stm_cfg.get("char_delay_ms", 50)
    delay_s  = delay_ms / 1000.0
    dry_run  = args.dry_run or stm_cfg.get("dry_run", False)
    cam_h    = args.cam_height if args.cam_height is not None else stm_cfg.get("cam_height_from_ref_mm", 390.0)

    print("\n" + "=" * 64)
    print("  Purple Detect → Depth → STM32 Coordinate Sender")
    print("=" * 64)
    print(f"  UART port      : {port}")
    print(f"  Char delay     : {delay_ms}ms")
    print(f"  Camera height  : {cam_h:.0f}mm ({cam_h/10:.0f}cm) from base")
    print(f"  Mode           : {'MANUAL (SPACE to send)' if args.manual else f'AUTO (every {args.send_interval}s)'}")
    print(f"  Depth          : {'DISABLED' if args.no_depth else 'ENABLED (stereo)'}")
    print(f"  Dry run        : {dry_run}")
    print("=" * 64)

    # ── Open UART ─────────────────────────────────────────────────────────────
    if dry_run:
        class FakeSerial:
            def write(self, data): pass
            def close(self): pass
        stm32 = FakeSerial()
        print(f"\n  ✓ UART: DRY-RUN mode")
    else:
        try:
            import serial
            stm32 = serial.Serial(port=port, baudrate=baud, timeout=1)
            print(f"\n  ✓ UART: connected to {port}")
        except Exception as e:
            print(f"\n  ✗ UART failed: {e}")
            print("    Try --dry-run to test without STM32")
            sys.exit(1)

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
            print(f"  ⚠ Depth engine failed: {e} — sending Z=0")

    # ── Main loop ─────────────────────────────────────────────────────────────
    cv2.namedWindow("Detect & Send | Press q=quit SPACE=send", cv2.WINDOW_NORMAL)

    print(f"\n  ✓ All systems ready!")
    print(f"  {'Press SPACE when purple is detected to send.' if args.manual else 'Auto-sending when purple detected.'}")
    print(f"  Press q to quit.\n")

    send_count = 0
    last_send_time = 0

    try:
        while True:
            left, right = read_stereo(cam_l, cam_r, W, H)

            # Apply software WB correction (removed as per preview.py handling)
            # left = correct_wb(left)

            # Keep preview on corrected frame for accurate colour representation.
            display_frame = left.copy()

            # ── Detect purple ─────────────────────────────────────────────
            cx, cy, area, mask = detect_purple(display_frame, cfg)

            # ── Compute depth ─────────────────────────────────────────────
            depth_mm = 0.0
            depth_diag = ""   # diagnostic string for display
            if cx is not None and depth_engine is not None:
                try:
                    dr = depth_engine.compute(left, right, (cx, cy), state="TEST")
                    if dr.depth_mm is not None:
                        depth_mm = dr.depth_mm
                        depth_diag = f"{depth_mm:.0f}mm (calibrated={dr.calibrated})"
                    elif dr.disparity_map is not None:
                        # Single-pixel disparity was 0 — try averaging a window
                        dmap = dr.disparity_map
                        win = 21  # window size
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
                            depth_diag = f"{depth_mm:.0f}mm (window median, {len(valid)} valid px)"
                        else:
                            depth_diag = f"ZERO disp in {win}x{win} window @ ({cx},{cy})"
                    else:
                        depth_diag = "No disparity map (ORB fallback failed)"
                except Exception as e:
                    depth_diag = f"Exception: {e}"

            # ── Build display ─────────────────────────────────────────────
            vis = display_frame.copy()

            if cx is not None:
                # Draw detection
                cv2.circle(vis, (cx, cy), 12, (0, 255, 0), -1)
                cv2.circle(vis, (cx, cy), 30, (255, 0, 255), 3)

                # ── STM32 coordinate mapping ───────────────────────────────────
                # X = horizontal angle (0=centre, +left, -right)
                # Y = depth from stereo (mm)
                # Z = height in arm coordinate system (0–325mm)
                send_x = 90.0 - ((cx / W) * 180.0)
                send_y = depth_mm

                # ── Z: Height in arm coordinate system ────────────────────────
                # Reference (0) is 390mm below camera centre.
                # Arm travels 0 → 325mm. Cannot go below 0.
                # Pinhole model: vertical offset = depth * (H/2 - cy) / focal
                #   cy < H/2 → grape above camera centre → higher Z
                #   cy > H/2 → grape below camera centre → lower Z
                CAM_HEIGHT_FROM_REF = 390.0
                ARM_MAX_MM = 325.0
                focal = depth_engine.focal_px if depth_engine else (H * 0.8)
                vert_offset = depth_mm * (H / 2.0 - cy) / focal if depth_mm > 0 else 0.0
                send_z = CAM_HEIGHT_FROM_REF + vert_offset - 115.0
                send_z = max(0.0, min(send_z, ARM_MAX_MM))

                # Info overlay
                info_lines = [
                    f"Purple @ pixel ({cx}, {cy})  area={area}px",
                    f"Depth: {depth_diag}" if depth_diag else "Depth: no engine",
                    f"STM32: X={send_x:.1f}  Y={send_y:.1f}  Z={send_z:.1f}",
                    f"(X=angle  Y=depth  Z=calibrated height)",
                ]
                for i, line in enumerate(info_lines):
                    cv2.putText(vis, line, (10, 30 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # ── Send to STM32 ─────────────────────────────────────────
                now = time.time()
                should_send = False

                if args.manual:
                    # Manual: handled by key press below
                    cv2.putText(vis, "SPACE to send", (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Auto: send if interval elapsed
                    if (now - last_send_time) >= args.send_interval:
                        should_send = True

                if should_send:
                    send_count += 1
                    print(f"\n  [#{send_count}] AUTO-SEND  "
                          f"pixel=({cx},{cy})  depth={depth_mm:.0f}mm  "
                          f"→ X={send_x:.1f} Y={send_y:.1f} Z={send_z:.1f}")
                    send_char_by_char(stm32, send_x, send_y, send_z, delay_s)
                    last_send_time = now
            else:
                cv2.putText(vis, "No purple detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show sends counter
            cv2.putText(vis, f"Sends: {send_count}", (W - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Detect & Send | Press q=quit SPACE=send", vis)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n  Quitting...")
                break

            elif key == ord(' ') and cx is not None:
                # Manual send on SPACE — same Z model as auto-send
                send_x = 90.0 - ((cx / W) * 180.0)
                send_y = depth_mm
                focal = depth_engine.focal_px if depth_engine else (H * 0.8)
                vert_offset = depth_mm * (H / 2.0 - cy) / focal if depth_mm > 0 else 0.0
                send_z = 390.0 + vert_offset - 115.0
                send_z = max(0.0, min(send_z, 325.0))
                send_count += 1
                print(f"\n  [#{send_count}] MANUAL SEND  "
                      f"pixel=({cx},{cy})  depth={depth_mm:.0f}mm  "
                      f"→ X={send_x:.1f} Y={send_y:.1f} Z={send_z:.1f}")
                send_char_by_char(stm32, send_x, send_y, send_z, delay_s)
                last_send_time = time.time()

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        print("\n  Shutting down...")
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cv2.destroyAllWindows()
        stm32.close()

        print(f"\n  Total coordinates sent: {send_count}")
        print("  ✓ All closed.\n")


if __name__ == "__main__":
    main()
