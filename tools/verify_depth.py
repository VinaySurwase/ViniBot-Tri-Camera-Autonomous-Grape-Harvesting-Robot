#!/usr/bin/env python3
"""
tools/verify_depth.py — Stereo Depth Accuracy Verifier
=======================================================
Place a flat target at known distances and press SPACE to capture.
The script computes depth estimates and compares against ground truth,
helping you verify baseline_mm and focal_length_px in config.yaml
before a real field run.

Usage:
    python tools/verify_depth.py
    python tools/verify_depth.py --distances 100 200 300 400 500
    python tools/verify_depth.py --point centre        # default
    python tools/verify_depth.py --point custom        # click to choose

Controls (live mode):
    SPACE     — capture and measure current pair
    r         — reset all measurements
    q         — quit and show summary
    Click     — set measurement point (in --point custom mode)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Depth verifier for GrapeBot stereo cameras.")
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--distances", type=float, nargs="+",
                   default=[100, 150, 200, 300, 400, 500],
                   help="Known ground-truth distances in mm")
    p.add_argument("--point", choices=["centre", "custom"], default="centre",
                   help="Where to measure depth (default: image centre)")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _fmt(v): return f"{v:.1f}" if v is not None else "N/A"

def _ok(m):   print(f"  \033[32m✓\033[0m  {m}")
def _warn(m): print(f"  \033[33m⚠\033[0m  {m}")
def _err(m):  print(f"  \033[31m✗\033[0m  {m}")


def open_pi_cameras(cfg: dict):
    """Open both Pi Camera V3 modules via picamera2."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        _err("picamera2 not installed. Run: pip install picamera2")
        sys.exit(1)

    W = cfg["cameras"]["stereo_left"]["width"]
    H = cfg["cameras"]["stereo_left"]["height"]

    def _open(idx, role):
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
        for _ in range(8): cam.capture_array()
        print(f"  ✓ {role} camera (index {idx}) ready")
        return cam

    L = _open(cfg["cameras"]["stereo_left"]["index"],  "LEFT")
    R = _open(cfg["cameras"]["stereo_right"]["index"], "RIGHT")
    return L, R, W, H


def read_stereo(cam_l, cam_r, out_w: int, out_h: int) -> tuple:
    rgb_l = cam_l.capture_array()
    rgb_r = cam_r.capture_array()
    # Keep as RGB (no conversion to BGR), exactly how preview.py handles it
    return cv2.resize(rgb_l, (out_w, out_h)), cv2.resize(rgb_r, (out_w, out_h))


def draw_crosshair(img: np.ndarray, x: int, y: int, colour=(0, 255, 255)) -> None:
    cv2.drawMarker(img, (x, y), colour, cv2.MARKER_CROSS, 30, 2)
    cv2.circle(img, (x, y), 10, colour, 2)


def draw_hud(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import logging
    logging.basicConfig(level=logging.WARNING)
    from depth.stereo_depth import StereoDepth

    engine = StereoDepth(cfg)

    print("\n" + "="*60)
    print("  GrapeBot Stereo Depth Verifier")
    print("="*60)
    print(f"  Calibrated : {engine._calibrated}")
    print(f"  Focal px   : {engine.focal_px:.1f}")
    print(f"  Baseline   : {engine.baseline_mm:.1f} mm")
    print(f"  Calib file : {engine.calib_file}")

    if not engine._calibrated:
        _warn("No calibration loaded — depth uses ORB fallback (less accurate).")
        _warn("Run: python calibration/run_calibration.py")
    print()

    cam_l, cam_r, W, H = open_pi_cameras(cfg)

    measurements: list[dict] = []
    dist_iter    = iter(args.distances)
    current_exp  = next(dist_iter, None)
    click_point  = (W // 2, H // 2)

    def _on_mouse(event, x, y, flags, _param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN and args.point == "custom":
            click_point = (x, y)

    cv2.namedWindow("Depth Verifier - LEFT | RIGHT", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Depth Verifier - LEFT | RIGHT", _on_mouse)
    cv2.resizeWindow("Depth Verifier - LEFT | RIGHT", 1280, 360)

    print(f"  Target distances: {args.distances} mm")
    print(f"  Place target at {current_exp} mm — press SPACE to capture")
    print("  Controls: SPACE=capture  r=reset  q=quit\n")

    try:
        while True:
            left, right = read_stereo(cam_l, cam_r, W, H)

            # Measurement point
            if args.point == "centre":
                px, py = W // 2, H // 2
            else:
                px, py = click_point

            # Build side-by-side canvas
            left_small  = cv2.resize(left,  (640, 360))
            right_small = cv2.resize(right, (640, 360))
            canvas      = np.hstack([left_small, right_small])

            # Scale point to small image
            spx = int(px * 640 / W)
            spy = int(py * 360 / H)
            draw_crosshair(canvas, spx,       spy)
            draw_crosshair(canvas, spx + 640, spy)

            # HUD
            exp_str = f"{current_exp:.0f}mm" if current_exp else "Done"
            hud     = (f"Expected: {exp_str}  |  Captured: {len(measurements)}  |  "
                       "SPACE=measure  r=reset  q=quit")
            if args.point == "custom":
                hud += "  |  Click to set point"

            if measurements:
                last = measurements[-1]
                err  = last["error_mm"]
                col  = (0, 255, 0) if abs(err) < 20 else ((0, 165, 255) if abs(err) < 50 else (0, 0, 255))
                cv2.putText(canvas,
                    f"Last: {last['measured']:.1f}mm  expected {last['expected']:.0f}mm  err={err:+.1f}mm",
                    (8, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            draw_hud(canvas, hud)
            cv2.imshow("Depth Verifier - LEFT | RIGHT", canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                measurements.clear()
                dist_iter   = iter(args.distances)
                current_exp = next(dist_iter, None)
                print("  Measurements reset.")

            elif key == ord(" "):
                dr = engine.compute(left, right, (px, py), state="VERIFY")

                if dr.depth_mm is None:
                    _warn("No depth estimate — reposition target or run calibration.")
                    continue

                error_mm  = dr.depth_mm - (current_exp or 0)
                error_pct = abs(error_mm) / (current_exp or 1) * 100
                cal_flag  = "[calibrated]" if dr.calibrated else "[fallback]"

                measurements.append({
                    "expected" : current_exp,
                    "measured" : dr.depth_mm,
                    "error_mm" : error_mm,
                    "error_pct": error_pct,
                    "latency"  : dr.latency_ms,
                })

                status = "✓" if abs(error_mm) < 30 else ("⚠" if abs(error_mm) < 80 else "✗")
                print(f"  {status}  Expected: {current_exp:>5.0f}mm  "
                      f"Measured: {dr.depth_mm:>7.1f}mm  "
                      f"Error: {error_mm:>+7.1f}mm ({error_pct:.1f}%)  "
                      f"{cal_flag}  {dr.latency_ms:.0f}ms")

                current_exp = next(dist_iter, None)
                if current_exp:
                    print(f"  → Place target at {current_exp:.0f}mm and press SPACE.")
                else:
                    print("  All distances measured. Press q to see summary.")

    finally:
        cam_l.stop(); cam_l.close()
        cam_r.stop(); cam_r.close()
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    if not measurements:
        print("\n  No measurements taken.")
        return

    print(f"\n{'='*60}")
    print("  DEPTH VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Expected':>10} {'Measured':>10} {'Error mm':>10} {'Error%':>8}  Status")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8}  {'─'*6}")

    abs_errors = []
    for m in measurements:
        status = "✓" if abs(m["error_mm"]) < 30 else ("⚠" if abs(m["error_mm"]) < 80 else "✗")
        print(f"  {m['expected']:>10.0f} {m['measured']:>10.1f} "
              f"{m['error_mm']:>+10.1f} {m['error_pct']:>8.1f}%  {status}")
        abs_errors.append(abs(m["error_mm"]))

    mean_err = sum(abs_errors) / len(abs_errors)
    print(f"\n  Mean absolute error : {mean_err:.1f} mm")

    if mean_err < 20:
        _ok("Calibration is excellent.")
    elif mean_err < 60:
        _warn(f"Moderate error ({mean_err:.1f}mm). Consider re-running calibration with 20+ pairs.")
    else:
        _err(f"High error ({mean_err:.1f}mm). Re-run calibration or check baseline_mm in config.yaml.")
        print("     Typical cause: baseline_mm in config does not match physical camera spacing.")
    print()


if __name__ == "__main__":
    main()