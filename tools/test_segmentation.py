#!/usr/bin/env python3
"""
tools/test_segmentation.py — Standalone Segmentation Tester
============================================================
Tests the YOLOv8n-seg NCNN segmentation model independently.
Loads the model exactly as test_subsystem.py does, runs inference,
extracts cutting points, and displays/saves annotated results.

Usage:
  python tools/test_segmentation.py --image test2.jpg              # single image
  python tools/test_segmentation.py --image test2.jpg --save       # save annotated output
  python tools/test_segmentation.py --live                          # live USB camera feed
  python tools/test_segmentation.py --live --save                   # live + save each frame
  python tools/test_segmentation.py --image test2.jpg --method centroid   # change cut method
  python tools/test_segmentation.py --image test2.jpg --conf 0.25  # raise confidence

Press 'q' to quit (in live mode), any key to advance (in image mode).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add project root so we can import segmentation, logging_, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Visualization helpers ────────────────────────────────────────────────────

# Per-stem colors so each mask gets a distinct overlay
COLORS = [
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (255, 165, 0),   # orange
    (255, 255, 0),   # yellow
    (128, 0, 255),   # violet
]


def draw_results(frame, result, cutting_result, cfg):
    """
    Draw segmentation masks, bounding boxes, scores, and cutting points
    on a copy of the frame. Returns the annotated image.
    """
    vis = frame.copy().astype(np.float32)

    # 1. Overlay masks with transparent color
    if result.masks is not None:
        for i, mask in enumerate(result.masks):
            color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
            vis[mask] = vis[mask] * 0.45 + color * 0.55

    vis = vis.astype(np.uint8)

    # 2. Draw bounding boxes + confidence scores
    if result.boxes is not None and result.scores is not None:
        for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
            x1, y1, x2, y2 = map(int, box)
            color = COLORS[i % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"stem {i}  {score:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 3. Draw cutting points
    if cutting_result and cutting_result.points:
        for cp in cutting_result.points:
            # Red cross at each cutting point
            cv2.drawMarker(vis, (cp.px, cp.py), (0, 0, 255),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, f"CUT ({cp.px},{cp.py})", (cp.px + 12, cp.py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Highlight the primary cutting point with a larger circle
        if cutting_result.primary:
            p = cutting_result.primary
            cv2.circle(vis, (p.px, p.py), 16, (0, 0, 255), 3)
            cv2.putText(vis, "PRIMARY", (p.px + 12, p.py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 4. Info bar at the top
    method = cfg["segmentation"].get("cutting_point_method", "topmost")
    info = (f"Stems: {result.stem_count}  |  "
            f"Cuts: {len(cutting_result.points) if cutting_result else 0}  |  "
            f"Method: {method}  |  "
            f"{result.latency_ms:.1f}ms")
    cv2.putText(vis, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


# ── USB camera helper ─────────────────────────────────────────────────────────

def open_usb_camera(cfg):
    """Open the USB segmentation camera as configured in config.yaml."""
    idx = cfg["cameras"]["usb_seg"]["index"]
    w   = cfg["cameras"]["usb_seg"].get("width", 1280)
    h   = cfg["cameras"]["usb_seg"].get("height", 720)

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
        print(f"  ✗ Failed to open USB camera at index {idx}")
        sys.exit(1)

    # Warm up — discard first few frames
    for _ in range(5):
        cap.read()

    print(f"  ✓ USB camera opened (index={idx}, {w}x{h})")
    return cap


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test YOLOv8n-seg NCNN segmentation model")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an input image (e.g. test2.jpg)")
    parser.add_argument("--live", action="store_true",
                        help="Use live USB camera feed")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to seg_output.jpg")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override confidence threshold (default: from config)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["topmost", "centroid", "midpoint"],
                        help="Cutting point method (default: from config)")
    args = parser.parse_args()

    if not args.image and not args.live:
        print("  ✗ Provide --image <path> or --live")
        parser.print_help()
        sys.exit(1)

    cfg = load_cfg()

    # Apply CLI overrides
    if args.conf is not None:
        cfg["segmentation"]["conf_thres"] = args.conf
    if args.method is not None:
        cfg["segmentation"]["cutting_point_method"] = args.method

    # ── Load segmentation model — same as test_subsystem.py ───────────────
    print("\n" + "=" * 56)
    print("  SEGMENTATION TEST")
    print("=" * 56)

    from segmentation.segmentor import Segmentor
    from segmentation.cutting_point import extract_cutting_points

    seg = Segmentor(cfg)
    if seg._model is None:
        print(f"  ✗ Model failed to load from: {cfg['segmentation']['model_path']}")
        sys.exit(1)

    seg_cfg = cfg["segmentation"]
    cut_method     = seg_cfg.get("cutting_point_method", "topmost")
    min_stem_area  = seg_cfg.get("min_stem_area_px", 50)

    print(f"  Model        : {seg_cfg['model_path']}")
    print(f"  Conf thresh  : {seg_cfg.get('conf_thres', 0.15)}")
    print(f"  Input size   : {seg_cfg.get('imgsz', 480)}")
    print(f"  Cut method   : {cut_method}")
    print(f"  Min stem area: {min_stem_area}px")
    print("=" * 56)

    # ── Single image mode ─────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"  ✗ Cannot load image: {args.image}")
            sys.exit(1)

        print(f"\n  Image: {args.image}  ({frame.shape[1]}x{frame.shape[0]})")
        print("  Running inference...\n")

        result = seg.segment(frame, state="TEST")
        cutting = None
        if result.success and result.masks is not None:
            cutting = extract_cutting_points(
                result.masks,
                method=cut_method,
                min_stem_area_px=min_stem_area,
            )

        # Print results
        print(f"  Success      : {result.success}")
        print(f"  Stems found  : {result.stem_count}")
        print(f"  Latency      : {result.latency_ms:.1f}ms")

        if result.boxes is not None:
            for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
                x1, y1, x2, y2 = map(int, box)
                print(f"  Stem {i}: box=({x1},{y1},{x2},{y2})  conf={score:.3f}")

        if cutting and cutting.points:
            print(f"\n  Cutting points ({cut_method}):")
            for cp in cutting.points:
                tag = " ← PRIMARY" if cutting.primary and cp == cutting.primary else ""
                print(f"    Stem {cp.stem_idx}: ({cp.px}, {cp.py})  "
                      f"area={cp.mask_area_px}px{tag}")
        elif result.success:
            print("  ⚠ No valid cutting points (masks too small?)")
        else:
            print(f"  ✗ Inference failed: {result.error or 'no masks returned'}")

        # Visualize
        vis = draw_results(frame, result, cutting, cfg)

        if args.save:
            out_path = "seg_output.jpg"
            cv2.imwrite(out_path, vis)
            print(f"\n  ✓ Saved annotated result → {out_path}")

        try:
            cv2.imshow("Segmentation Test", vis)
            print("\n  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            if not args.save:
                cv2.imwrite("seg_output.jpg", vis)
                print("  (No display available — saved to seg_output.jpg)")

    # ── Live USB camera mode ──────────────────────────────────────────────
    elif args.live:
        cap = open_usb_camera(cfg)
        frame_count = 0
        total_ms = 0.0

        print("\n  Live mode — press 'q' to quit, 's' to save current frame.\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("  ⚠ Frame read failed, retrying...")
                    continue

                result = seg.segment(frame, state="TEST")
                cutting = None
                if result.success and result.masks is not None:
                    cutting = extract_cutting_points(
                        result.masks,
                        method=cut_method,
                        min_stem_area_px=min_stem_area,
                    )

                frame_count += 1
                total_ms += result.latency_ms

                # Print every 10th frame to avoid flooding terminal
                if frame_count % 10 == 0:
                    avg = total_ms / frame_count
                    cuts = len(cutting.points) if cutting else 0
                    print(f"  Frame {frame_count:>4}: stems={result.stem_count}  "
                          f"cuts={cuts}  {result.latency_ms:.1f}ms  "
                          f"avg={avg:.1f}ms  ({1000/avg:.1f} FPS)")

                vis = draw_results(frame, result, cutting, cfg)
                cv2.imshow("Segmentation Live | q=quit s=save", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    out = f"seg_live_{frame_count}.jpg"
                    cv2.imwrite(out, vis)
                    print(f"  ✓ Saved → {out}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if frame_count > 0:
                avg = total_ms / frame_count
                print(f"\n  Total frames : {frame_count}")
                print(f"  Avg latency  : {avg:.1f}ms  ({1000/avg:.1f} FPS)")
            print("  ✓ Done.\n")


if __name__ == "__main__":
    main()
