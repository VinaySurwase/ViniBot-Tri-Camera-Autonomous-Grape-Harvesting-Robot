#!/usr/bin/env python3
"""
tools/test_purple_distance.py
=============================
Shows live LEFT + RIGHT camera feed.
Highlights the largest purple object.
Press SPACE to capture one frame and compute its distance.
Press 'q' to quit.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import yaml


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def open_pi_cameras(cfg):
    try:
        from picamera2 import Picamera2
        from libcamera import controls as libcam_controls
    except ImportError:
        print("picamera2 not installed.")
        sys.exit(1)

    W = cfg["cameras"]["stereo_left"]["width"]
    H = cfg["cameras"]["stereo_left"]["height"]

    def _open(idx, role):
        print(f"Opening {role} camera...")
        cam = Picamera2(idx)
        config = cam.create_preview_configuration(
            main={"size": (2304, 1296), "format": "RGB888"}
        )
        config["controls"] = {
            "FrameRate": 20.0,
        }
        cam.configure(config)
        cam.start()
        props = cam.camera_properties
        full_size = props.get("PixelArraySize", None)
        if full_size is not None and len(full_size) == 2:
            cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})

        # Give the ISP 2 seconds to converge AWB + AE (this is the key fix!)
        import time
        time.sleep(2.0)
        for _ in range(15):
            cam.capture_array()

        print(
            f"  ✓ {role} camera ready"
        )
        return cam

    L = _open(cfg["cameras"]["stereo_left"]["index"], "LEFT")
    R = _open(cfg["cameras"]["stereo_right"]["index"], "RIGHT")
    return L, R, W, H


def read_stereo(cam_l, cam_r, out_w, out_h):
    rgb_l = cam_l.capture_array()
    rgb_r = cam_r.capture_array()
    # Keep as RGB (no conversion to BGR), exactly how preview.py handles it
    return cv2.resize(rgb_l, (out_w, out_h)), cv2.resize(rgb_r, (out_w, out_h))


def main():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg = load_cfg(cfg_path)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from depth.stereo_depth import StereoDepth

    engine = StereoDepth(cfg)
    cam_l, cam_r, W, H = open_pi_cameras(cfg)

    # Hardcoded stronger purple thresholds (bypassing config)
    LOWER_PURPLE = np.array([130, 80, 80])
    UPPER_PURPLE = np.array([165, 255, 255])

    cv2.namedWindow("Purple Tracker | LEFT | DISPARITY (Depth) MAP", cv2.WINDOW_NORMAL)

    print("\n" + "=" * 55)
    print("  Purple Distance Tester")
    print("=" * 55)
    print(f"  Loaded Focal Length : {engine.focal_px:.1f} px")
    print(f"  Loaded Baseline     : {engine.baseline_mm:.1f} mm")
    print("=" * 55)
    print("  SPACE = capture & measure distance")
    print("  q     = quit")
    print("=" * 55 + "\n")

    capture_count = 0

    try:
        while True:
            left, right = read_stereo(cam_l, cam_r, W, H)
            
            # --- CRITICAL FIX: Track purple in the RECTIFIED view! ---
            # If we track in the raw view, the (cx, cy) coordinate won't match 
            # the depth map because the stereo engine warps the image.
            left_rectified = engine.rectify_left(left)

            # Detect purple in the rectified frame
            hsv = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_PURPLE, UPPER_PURPLE)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display_left = left_rectified.copy()
            cx, cy = None, None

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 500:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(display_left, [largest], -1, (255, 0, 255), 3)
                        cv2.circle(display_left, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.putText(display_left, "Purple detected — press SPACE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_left, "No purple detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Compute depth continuously so we can visualize the disparity map live
            # Default to checking center of screen if no purple found
            check_x = cx if cx is not None else W // 2
            check_y = cy if cy is not None else H // 2
            
            try:
                dr = engine.compute(left, right, (check_x, check_y), state="TEST")
                disp_map = dr.disparity_map
            except Exception:
                dr = None
                disp_map = None

            # Render the disparity map side-by-side
            if disp_map is not None:
                disp_vis = cv2.normalize(disp_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                disp_vis = np.uint8(disp_vis)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                
                # Draw where we are looking on the depth map
                if cx is not None and cy is not None:
                    # White circle with black crosshair
                    cv2.circle(disp_color, (cx, cy), 10, (255, 255, 255), 2)
                    cv2.drawMarker(disp_color, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)
            else:
                disp_color = np.zeros_like(display_left)

            combined = np.hstack([display_left, disp_color])
            combined = cv2.resize(combined, (1280, 480))
            cv2.imshow("Purple Tracker | LEFT | DISPARITY (Depth) MAP", combined)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n  Quitting...\n")
                break

            elif key == ord(' '):
                capture_count += 1
                print(f"\n  --- Capture #{capture_count} ---")

                if cx is None or cy is None:
                    print("  ✗ No purple object found in frame. Point the camera at something purple.\n")
                    continue

                print(f"  Purple center : ({cx}, {cy})")

                if dr is not None and dr.depth_mm is not None:
                    print(f"  ✓ Distance    : {dr.depth_mm:.1f} mm  ({dr.depth_mm / 10:.1f} cm)")
                else:
                    print("  ✗ Could not compute distance (disparity was zero/negative).")
                print()

    finally:
        print("Shutting down cameras...")
        cam_l.stop()
        cam_l.close()
        cam_r.stop()
        cam_r.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
