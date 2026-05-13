#!/usr/bin/env python3
"""
calibration/run_calibration.py — Stereo Camera Calibration
===========================================================
Captures chessboard pairs from both Pi Camera V3 modules using picamera2,
computes stereo rectification maps, and saves to calibration/stereo_calib.npz.

The saved file is loaded by depth/stereo_depth.py at runtime.

Usage:
    python calibration/run_calibration.py
    python calibration/run_calibration.py --cols 9 --rows 6 --square 25.0
    python calibration/run_calibration.py --left 0 --right 1 --min-pairs 20

Controls during capture:
    SPACE  — capture current stereo pair (if chessboard found in BOTH cameras)
    d      — delete last captured pair
    c      — calibrate now (if enough pairs)
    q      — quit without calibrating

Requirements:
    - Pi Camera V3 modules at index 0 (left/detection) and 1 (right/detection)
    - Printed chessboard: 9×6 inner corners, square size = your measured mm
    - Good lighting, no motion blur
    - At least 15 pairs from varied angles/distances
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_COLS       = 9        # inner corners horizontal
DEFAULT_ROWS       = 5        # inner corners vertical
DEFAULT_SQUARE_MM  = 24.0     # physical square size in mm
DEFAULT_MIN_PAIRS  = 15
DEFAULT_LEFT_IDX   = 0        # left Pi cam (stereo_left / detection)
DEFAULT_RIGHT_IDX  = 1        # right Pi cam (stereo_right / detection)
DEFAULT_WIDTH      = 1280
DEFAULT_HEIGHT     = 720
DEFAULT_OUTPUT     = "calibration/stereo_calib.npz"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo calibration for GrapeBot Pi cameras.")
    p.add_argument("--cols",      type=int,   default=DEFAULT_COLS,      help="Inner corners cols")
    p.add_argument("--rows",      type=int,   default=DEFAULT_ROWS,      help="Inner corners rows")
    p.add_argument("--square",    type=float, default=DEFAULT_SQUARE_MM, help="Square size (mm)")
    p.add_argument("--min-pairs", type=int,   default=DEFAULT_MIN_PAIRS, help="Min pairs to calibrate")
    p.add_argument("--left",      type=int,   default=DEFAULT_LEFT_IDX,  help="Left camera index")
    p.add_argument("--right",     type=int,   default=DEFAULT_RIGHT_IDX, help="Right camera index")
    p.add_argument("--width",     type=int,   default=DEFAULT_WIDTH)
    p.add_argument("--height",    type=int,   default=DEFAULT_HEIGHT)
    p.add_argument("--output",    default=DEFAULT_OUTPUT, help="Output .npz path")
    return p.parse_args()


def open_cameras(args: argparse.Namespace):
    """Open both Pi cameras with picamera2. Returns (cam_left, cam_right)."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not installed. Run: pip install picamera2")
        sys.exit(1)

    def _open(idx: int, role: str):
        print(f"  Opening {role} camera (index {idx})…")
        cam = Picamera2(idx)
        cfg = cam.create_preview_configuration(
            main={"size": (2304, 1296), "format": "BGR888"}
        )
        cfg["controls"] = {
            "FrameRate": 20.0,
            "AwbEnable": False,
            "ColourGains": (1.83, 1.60),   # calibrated to match rpicam-hello
        }
        cam.configure(cfg)
        cam.start()
        time.sleep(0.5)
        # Warmup
        for _ in range(8):
            cam.capture_array()
        print(f"  {role} camera ready.")
        return cam

    cam_l = _open(args.left,  "LEFT (stereo_left)")
    cam_r = _open(args.right, "RIGHT (stereo_right)")
    return cam_l, cam_r


def capture_frame(cam, label: str, img_size: tuple = (1280, 720)) -> np.ndarray:
    """Capture one BGR frame from a picamera2 instance."""
    bgr = cam.capture_array()
    # Resize from 2304x1296 back to the requested calibration size
    return cv2.resize(bgr, img_size)


def run_calibration(
    obj_pts: list,
    img_pts_l: list,
    img_pts_r: list,
    img_size: tuple,
    output_path: str,
) -> bool:
    """Run stereo calibration and save results. Returns True on success."""
    print(f"\n  Running individual camera intrinsic calibrations…")
    
    # Calibrate each camera independently first (crucial for stability)
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_pts, img_pts_l, img_size, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_pts, img_pts_r, img_size, None, None)
    
    print(f"  Left Camera intrinsic RMS  : {ret_l:.2f} px")
    print(f"  Right Camera intrinsic RMS : {ret_r:.2f} px")

    print(f"  Running stereo calibration with {len(obj_pts)} pairs…")

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_SAME_FOCAL_LENGTH
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)

    rms, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_l, img_pts_r,
        K_l, D_l, K_r, D_r,
        img_size, flags=flags, criteria=criteria,
    )

    print(f"  Stereo reprojection RMS: {rms:.4f} px")
    quality = "✓ Excellent" if rms < 0.5 else ("✓ Good" if rms < 1.5 else "⚠ High — recapture more pairs")
    print(f"  Quality                : {quality}")

    if rms > 3.0:
        print("  ✗ RMS > 3.0 — calibration rejected. Capture more diverse pairs or check swap.")
        return False

    # Stereo rectification
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        KL, DL, KR, DR, img_size, R, T, alpha=0, newImageSize=img_size
    )

    map1L, map2L = cv2.initUndistortRectifyMap(KL, DL, RL, PL, img_size, cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(KR, DR, RR, PR, img_size, cv2.CV_16SC2)

    focal_px    = float(PL[0, 0])
    baseline_mm = float(abs(T[0, 0]))   # T[0] = horizontal baseline in mm

    print(f"  Focal length           : {focal_px:.1f} px")
    print(f"  Baseline               : {baseline_mm:.2f} mm")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        KL=KL, DL=DL, KR=KR, DR=DR,
        R=R, T=T, E=E, F=F,
        RL=RL, RR=RR, PL=PL, PR=PR, Q=Q,
        map1L=map1L, map2L=map2L,
        map1R=map1R, map2R=map2R,
        focal_px=np.float32(focal_px),
        baseline_mm=np.float32(baseline_mm),
        rms=np.float32(rms),
    )

    print(f"  ✅ Saved → {output_path}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    COLS  = args.cols
    ROWS  = args.rows
    SQ    = args.square
    PATTERN = (COLS, ROWS)

    # 3-D world points for one chessboard view
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQ

    obj_pts:   list = []
    img_pts_l: list = []
    img_pts_r: list = []
    img_size:  tuple = (args.width, args.height)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "="*60)
    print("  GrapeBot Stereo Calibration")
    print("="*60)
    print(f"  Pattern  : {COLS}×{ROWS} inner corners")
    print(f"  Square   : {SQ} mm")
    print(f"  Cameras  : left={args.left}, right={args.right}")
    print(f"  Min pairs: {args.min_pairs}")
    print(f"  Output   : {args.output}")
    print()
    print("  Controls: SPACE=capture  d=delete last  c=calibrate  q=quit")
    print()

    cam_l, cam_r = open_cameras(args)

    cv2.namedWindow("LEFT",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("RIGHT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LEFT",  640, 360)
    cv2.resizeWindow("RIGHT", 640, 360)

    try:
        while True:
            frame_l = capture_frame(cam_l, "LEFT", img_size)
            frame_r = capture_frame(cam_r, "RIGHT", img_size)

            grey_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            grey_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            found_l, corners_l = cv2.findChessboardCorners(grey_l, PATTERN)
            found_r, corners_r = cv2.findChessboardCorners(grey_r, PATTERN)

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()
            cv2.drawChessboardCorners(disp_l, PATTERN, corners_l, found_l)
            cv2.drawChessboardCorners(disp_r, PATTERN, corners_r, found_r)

            n = len(obj_pts)
            status = (
                f"Pairs: {n}/{args.min_pairs}  |  "
                f"L={'✓' if found_l else '✗'}  R={'✓' if found_r else '✗'}  |  "
                "SPACE=capture  d=delete  c=calibrate  q=quit"
            )

            for img in (disp_l, disp_r):
                cv2.putText(img, status, (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(img, f"Pairs captured: {n}", (8, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if n >= args.min_pairs else (0, 140, 255), 2)

            cv2.imshow("LEFT",  cv2.resize(disp_l, (640, 360)))
            cv2.imshow("RIGHT", cv2.resize(disp_r, (640, 360)))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("  Quit without calibrating.")
                break

            elif key == ord(" "):
                if found_l and found_r:
                    c_l = cv2.cornerSubPix(grey_l, corners_l, (11, 11), (-1, -1), criteria)
                    c_r = cv2.cornerSubPix(grey_r, corners_r, (11, 11), (-1, -1), criteria)
                    
                    # CRITICAL FIX: Ensure both cameras count corners in the exact same direction
                    # Since cameras are side-by-side, the Y-coordinates of Corner 0 must be similar.
                    y_diff_normal = abs(c_l[0][0][1] - c_r[0][0][1])
                    y_diff_inverted = abs(c_l[0][0][1] - c_r[-1][0][1])
                    
                    if y_diff_inverted < y_diff_normal:
                        print("    [Auto-Fix] Right pattern was detected backwards! Flipped to match Left.")
                        c_r = c_r[::-1]  # Reverse the array so point 0 matches point 0

                    obj_pts.append(objp)
                    img_pts_l.append(c_l)
                    img_pts_r.append(c_r)
                    print(f"  ✓ Pair {len(obj_pts)} captured.")
                else:
                    print("  ✗ Chessboard not found in both cameras. Reposition and try again.")

            elif key == ord("d"):
                if obj_pts:
                    obj_pts.pop(); img_pts_l.pop(); img_pts_r.pop()
                    print(f"  Deleted last pair. Remaining: {len(obj_pts)}")

            elif key == ord("c"):
                if len(obj_pts) < args.min_pairs:
                    print(f"  Need at least {args.min_pairs} pairs (have {len(obj_pts)}).")
                else:
                    ok = run_calibration(obj_pts, img_pts_l, img_pts_r,
                                         img_size, args.output)
                    if ok:
                        break

    finally:
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cv2.destroyAllWindows()

    # Auto-calibrate if enough pairs collected and user quit with q
    if len(obj_pts) >= args.min_pairs and not Path(args.output).exists():
        run_calibration(obj_pts, img_pts_l, img_pts_r, img_size, args.output)


if __name__ == "__main__":
    main()