#!/usr/bin/env python3
"""
tools/tune_hsv.py — HSV Threshold Tuner v2
===========================================
Live trackbar tuner for grape detection thresholds.
Supports picamera2 (Pi Camera V3) and USB camera.
Saves tuned values directly to config/config.yaml.

Usage:
    python tools/tune_hsv.py                     # picamera2 index 0
    python tools/tune_hsv.py --source usb        # USB camera
    python tools/tune_hsv.py --image frame.jpg   # static image
    python tools/tune_hsv.py --camera 1          # Pi cam index 1

Controls:
    s  — save to config.yaml
    r  — reset to config.yaml values
    c  — toggle HoughCircles overlay
    q  — quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

WIN_CTRL = "HSV Controls"
WIN_VIEW = "Live + Mask"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--source", choices=["picamera2", "usb"], default="picamera2")
    p.add_argument("--camera", default=0)
    p.add_argument("--image",  default=None)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


import re

def save_cfg(cfg, path, tb):
    with open(path, "r") as f:
        content = f.read()

    # Regex replacements
    replacements = {
        r"(hsv_lower:\s*\[).*?(\])": f"\\g<1>{tb['h_min']}, {tb['s_min']}, {tb['v_min']}\\g<2>",
        r"(hsv_upper:\s*\[).*?(\])": f"\\g<1>{tb['h_max']}, {tb['s_max']}, {tb['v_max']}\\g<2>",
        r"(morph_kernel_size:\s*)\d+": f"\\g<1>{tb['morph']}",
        r"(min_contour_area:\s*)\d+": f"\\g<1>{tb['min_area']}",
        r"(hough_param2:\s*)\d+": f"\\g<1>{tb['hough_p2']}",
        r"(min_circle_radius:\s*)\d+": f"\\g<1>{tb['min_r']}",
        r"(max_circle_radius:\s*)\d+": f"\\g<1>{tb['max_r']}"
    }

    for pattern, repl in replacements.items():
        content = re.sub(pattern, repl, content)

    with open(path, "w") as f:
        f.write(content)
        
    print(f"  ✓ Saved → {path} (comments preserved)")


def make_trackbars(cfg):
    d  = cfg["detection"]
    lo = d["hsv_lower"]
    hi = d["hsv_upper"]
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 360)
    for name, val, maxv in [
        ("H min", lo[0], 179), ("S min", lo[1], 255), ("V min", lo[2], 255),
        ("H max", hi[0], 179), ("S max", hi[1], 255), ("V max", hi[2], 255),
        ("Morph K",    d.get("morph_kernel_size", 5),  15),
        ("Min area",   d.get("min_contour_area",  300), 3000),
        ("Hough p2",   d.get("hough_param2",       30), 100),
        ("Min r(px)",  d.get("min_circle_radius",   8),  80),
        ("Max r(px)",  d.get("max_circle_radius",  60), 150),
    ]:
        cv2.createTrackbar(name, WIN_CTRL, val, maxv, lambda _: None)


def read_tb():
    def g(n): return cv2.getTrackbarPos(n, WIN_CTRL)
    return {
        "h_min": g("H min"), "s_min": g("S min"), "v_min": g("V min"),
        "h_max": g("H max"), "s_max": g("S max"), "v_max": g("V max"),
        "morph": max(1, g("Morph K")),
        "min_area": g("Min area"),
        "hough_p2": g("Hough p2"),
        "min_r": g("Min r(px)"),
        "max_r": g("Max r(px)"),
    }


def process(frame, tb, show_circles):
    lower = np.array([tb["h_min"], tb["s_min"], tb["v_min"]], dtype=np.uint8)
    upper = np.array([tb["h_max"], tb["s_max"], tb["v_max"]], dtype=np.uint8)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = tb["morph"]
    kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= tb["min_area"]:
            cv2.drawContours(clean, [c], -1, 255, -1)

    vis = frame.copy()
    green        = np.zeros_like(frame)
    green[:, :, 1] = clean
    vis = cv2.addWeighted(vis, 0.7, green, 0.3, 0)

    circles = []
    if show_circles and clean.sum() > 0 and tb["hough_p2"] > 0:
        grey = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), clean)
        grey = cv2.GaussianBlur(grey, (9, 9), 2)
        raw  = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1.2, 30,
                                 param1=50, param2=tb["hough_p2"],
                                 minRadius=tb["min_r"], maxRadius=tb["max_r"])
        if raw is not None:
            for x, y, r in np.round(raw[0]).astype(int):
                cv2.circle(vis, (x, y), r, (0, 200, 255), 2)
                cv2.circle(vis, (x, y), 2, (0, 200, 255), -1)
                circles.append((x, y, r))

    info = [
        f"HSV lower [{tb['h_min']},{tb['s_min']},{tb['v_min']}]  upper [{tb['h_max']},{tb['s_max']},{tb['v_max']}]",
        f"Mask px:{clean.sum()//255:,}   Circles:{len(circles)}",
        "s=save  r=reset  c=circles  q=quit",
    ]
    for i, line in enumerate(info):
        cv2.putText(vis, line, (8, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([cv2.resize(vis, (640, 360)),
                          cv2.resize(mask_bgr, (640, 360))])
    return combined


def open_source(args):
    cam_id = int(args.camera) if str(args.camera).isdigit() else args.camera
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: cannot load {args.image}")
            sys.exit(1)
        return "image", img, None

    if args.source == "picamera2":
        try:
            from picamera2 import Picamera2
            cam = Picamera2(cam_id)
            config = cam.create_preview_configuration(
                main={"size": (2304, 1296), "format": "RGB888"}
            )
            config["controls"] = {
                "FrameRate": 30.0,
            }
            cam.configure(config)
            cam.start()
            props = cam.camera_properties
            full_size = props.get("PixelArraySize", None)
            if full_size is not None and len(full_size) == 2:
                cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})
            for _ in range(25): cam.capture_array()
            return "picamera2", None, cam
        except Exception as e:
            print(f"Picamera2 failed: {e} — falling back to USB.")

    if isinstance(cam_id, str) and cam_id.startswith('/dev/'):
        import os
        real_idx = os.path.realpath(cam_id)
        if real_idx.startswith('/dev/video'):
            try:
                cam_id = int(real_idx.replace('/dev/video', ''))
            except ValueError:
                pass
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(5): cap.read()
    return "usb", None, cap


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)
    make_trackbars(cfg)
    cv2.namedWindow(WIN_VIEW, cv2.WINDOW_NORMAL)

    src_type, static_img, cam = open_source(args)
    print(f"\n  Source: {src_type}  —  s=save  r=reset  c=circles  q=quit\n")
    show_circles = True

    while True:
        if src_type == "image":
            frame = static_img.copy()
        elif src_type == "picamera2":
            frame = cam.capture_array()
            frame = cv2.resize(frame, (1280, 720))
        else:
            ok, frame = cam.read()
            if not ok: continue

        tb  = read_tb()
        out = process(frame, tb, show_circles)
        cv2.imshow(WIN_VIEW, out)

        key = cv2.waitKey(50 if src_type == "image" else 30) & 0xFF
        if   key == ord("q"): break
        elif key == ord("s"): save_cfg(cfg, args.config, tb)
        elif key == ord("r"):
            cfg = load_cfg(args.config); make_trackbars(cfg)
            print("  Trackbars reset.")
        elif key == ord("c"):
            show_circles = not show_circles
            print(f"  Circles: {'ON' if show_circles else 'OFF'}")

    if src_type == "picamera2" and cam:
        cam.stop(); cam.close()
    elif src_type == "usb" and cam:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()