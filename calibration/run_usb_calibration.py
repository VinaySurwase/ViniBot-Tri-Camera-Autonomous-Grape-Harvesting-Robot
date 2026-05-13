#!/usr/bin/env python3
"""
calibration/run_usb_calibration.py — USB Camera Intrinsic Calibration
=====================================================================
Captures chessboard images from the USB camera to compute its intrinsic matrix 
and exact focal length. This is useful for precise angle calculations for the end-effector.

Usage:
    python calibration/run_usb_calibration.py
    python calibration/run_usb_calibration.py --cols 9 --rows 6 --square 25.0

Controls during capture:
    SPACE  — capture current frame (if chessboard found)
    d      — delete last captured frame
    c      — calibrate now (requires at least 10 frames)
    q      — quit without calibrating
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_COLS       = 9        # inner corners horizontal
DEFAULT_ROWS       = 5        # inner corners vertical
DEFAULT_SQUARE_MM  = 24.0     # physical square size in mm
DEFAULT_MIN_FRAMES = 10
DEFAULT_OUTPUT     = "calibration/usb_calib.npz"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intrinsic calibration for USB Camera.")
    p.add_argument("--cols",      type=int,   default=DEFAULT_COLS,      help="Inner corners cols")
    p.add_argument("--rows",      type=int,   default=DEFAULT_ROWS,      help="Inner corners rows")
    p.add_argument("--square",    type=float, default=DEFAULT_SQUARE_MM, help="Square size (mm)")
    p.add_argument("--min-frames",type=int,   default=DEFAULT_MIN_FRAMES,help="Min frames to calibrate")
    p.add_argument("--output",    default=DEFAULT_OUTPUT, help="Output .npz path")
    return p.parse_args()


def load_cfg():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run_calibration(
    obj_pts: list,
    img_pts: list,
    img_size: tuple,
    output_path: str,
) -> bool:
    """Run intrinsic calibration and save results. Returns True on success."""
    print(f"\n  Running USB Camera intrinsic calibration with {len(obj_pts)} frames…")
    
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    
    print(f"  Camera intrinsic RMS: {ret:.2f} px")
    
    if ret > 3.0:
        print("  ✗ RMS > 3.0 — calibration rejected. Capture more diverse frames.")
        return False

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print("\n  ==== CALIBRATION RESULTS ====")
    print(f"  Focal Length (X): {fx:.2f} px")
    print(f"  Focal Length (Y): {fy:.2f} px")
    print(f"  Principal Point : ({cx:.1f}, {cy:.1f})")
    print("  =============================\n")
    print("  Use this Focal Length (Y) in test_endeffector.py to compute the exact angle:")
    print("  angle = math.degrees(math.atan((cy_target - principal_y) / focal_y))")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        K=K, D=D,
        fx=np.float32(fx),
        fy=np.float32(fy),
        cx=np.float32(cx),
        cy=np.float32(cy),
        rms=np.float32(ret),
    )

    print(f"  ✅ Saved → {output_path}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    cfg   = load_cfg()
    
    COLS  = args.cols
    ROWS  = args.rows
    SQ    = args.square
    PATTERN = (COLS, ROWS)

    # 3-D world points for one chessboard view
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQ

    obj_pts:   list = []
    img_pts:   list = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "="*60)
    print("  GrapeBot USB Camera Calibration")
    print("="*60)
    print(f"  Pattern  : {COLS}×{ROWS} inner corners")
    print(f"  Square   : {SQ} mm")
    print(f"  Min frames: {args.min_frames}")
    print(f"  Output   : {args.output}")
    print()
    print("  Controls: SPACE=capture  d=delete last  c=calibrate  q=quit")
    print()

    # Setup USB Camera
    idx = cfg["cameras"]["usb_seg"]["index"]
    w   = cfg["cameras"]["usb_seg"].get("width", 1280)
    h   = cfg["cameras"]["usb_seg"].get("height", 720)
    img_size = (w, h)

    # Resolve symlink if needed
    if isinstance(idx, str) and idx.startswith('/dev/'):
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

    cv2.namedWindow("USB CAMERA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("USB CAMERA", 800, 450)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                time.sleep(0.5)
                continue

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(grey, PATTERN)

            disp = frame.copy()
            cv2.drawChessboardCorners(disp, PATTERN, corners, found)

            n = len(obj_pts)
            status = (
                f"Frames: {n}/{args.min_frames}  |  "
                f"Chessboard: {'✓' if found else '✗'}  |  "
                "SPACE=capture  d=delete  c=calibrate  q=quit"
            )

            cv2.putText(disp, status, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(disp, f"Captured: {n}", (8, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if n >= args.min_frames else (0, 140, 255), 2)

            cv2.imshow("USB CAMERA", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("  Quit without calibrating.")
                break

            elif key == ord(" "):
                if found:
                    c_refined = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
                    obj_pts.append(objp)
                    img_pts.append(c_refined)
                    print(f"  ✓ Frame {len(obj_pts)} captured.")
                else:
                    print("  ✗ Chessboard not found. Reposition and try again.")

            elif key == ord("d"):
                if obj_pts:
                    obj_pts.pop(); img_pts.pop()
                    print(f"  Deleted last frame. Remaining: {len(obj_pts)}")

            elif key == ord("c"):
                if len(obj_pts) < args.min_frames:
                    print(f"  Need at least {args.min_frames} frames (have {len(obj_pts)}).")
                else:
                    ok = run_calibration(obj_pts, img_pts, img_size, args.output)
                    if ok:
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
