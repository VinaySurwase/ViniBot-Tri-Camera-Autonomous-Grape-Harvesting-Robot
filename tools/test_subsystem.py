#!/usr/bin/env python3
"""
tools/test_subsystem.py — Subsystem Test Runner v2
===================================================
Tests each module independently before running the full robot.

Usage:
    python tools/test_subsystem.py cameras    --frames 20 --show
    python tools/test_subsystem.py detection  --image frame.jpg --show
    python tools/test_subsystem.py detection  --frames 30 --show
    python tools/test_subsystem.py seg        --image frame.jpg --show
    python tools/test_subsystem.py depth      --frames 10 --show
    python tools/test_subsystem.py coords     --dry-run
    python tools/test_subsystem.py stm32      --dry-run
    python tools/test_subsystem.py all        --image frame.jpg
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path so modules like 'core' can be found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def _ok(m):   print(f"  \033[32m✓\033[0m  {m}")
def _warn(m): print(f"  \033[33m⚠\033[0m  {m}")
def _err(m):  print(f"  \033[31m✗\033[0m  {m}")
def _title(t):print(f"\n\033[36m{'='*56}\n  {t}\n{'='*56}\033[0m")


# ── Camera helpers ────────────────────────────────────────────────────────────

def _get_pi_frame(index=0, out_w=1280, out_h=720):
    from picamera2 import Picamera2
    cam = Picamera2(index)
    # Request exactly 2304x1296 (the native full FOV mode of IMX708). 
    # We downscale to 1280x720 in Python to bypass libcamera cropping bugs.
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
    for _ in range(30): cam.capture_array()
    # Keep as RGB (no conversion to BGR), exactly how preview.py handles it
    frame = cam.capture_array()
    frame = cv2.resize(frame, (out_w, out_h))
    cam.stop(); cam.close()
    return frame


def _get_usb_frame(index=0, cfg=None):
    idx = cfg["cameras"]["usb_seg"]["index"] if cfg else index
    if isinstance(idx, str) and idx.startswith('/dev/'):
        import os
        real_idx = os.path.realpath(idx)
        if real_idx.startswith('/dev/video'):
            try:
                idx = int(real_idx.replace('/dev/video', ''))
            except ValueError:
                pass
    cap = cv2.VideoCapture(idx)
    for _ in range(5): cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _frame_from_args(args, cfg, cam_type="left"):
    if args.image:
        f = cv2.imread(args.image)
        if f is None:
            _err(f"Cannot load {args.image}"); sys.exit(1)
        return f
    if cam_type in ("left", "right"):
        cam_cfg = cfg["cameras"]["stereo_left" if cam_type == "left" else "stereo_right"]
        idx = cam_cfg["index"]
        return _get_pi_frame(idx, cam_cfg["width"], cam_cfg["height"])
    return _get_usb_frame(cfg=cfg)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cameras(args, cfg):
    _title("CAMERA TEST — All 3 Cameras")
    from core.camera_manager import CameraManager
    cam = CameraManager(cfg)
    ok  = cam.open_all()
    if not ok:
        _err("One or more cameras failed to open."); return False

    det_ok = seg_ok = 0
    t0 = time.time()
    for i in range(args.frames):
        ok_s, L, R = cam.read_stereo()
        ok_u, U    = cam.read_seg()
        if ok_s: det_ok += 1
        if ok_u: seg_ok += 1
        if ok_s and ok_u:
            row1 = np.hstack([cv2.resize(L,(480,270)), cv2.resize(R,(480,270))])
            row2 = cv2.resize(U, (960, 270))
            combined = np.vstack([row1, row2])
            cv2.imwrite("test_cameras.jpg", combined)  # Always save a frame for headless inspection
            
            if args.show:
                try:
                    cv2.imshow("Left|Right  /  USB", combined)
                    if cv2.waitKey(30) & 0xFF == ord("q"): break
                except Exception as e:
                    pass # Display failed (likely SSH), but we saved test_cameras.jpg

    elapsed = time.time() - t0
    cam.close_all()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    _ok(f"Stereo frames: {det_ok}/{args.frames}  ({det_ok/elapsed:.1f} FPS)")
    _ok(f"USB frames:    {seg_ok}/{args.frames}  ({seg_ok/elapsed:.1f} FPS)")
    if det_ok < args.frames * 0.9 or seg_ok < args.frames * 0.9:
        _warn("Drop rate > 10% — check connections.")
    return True


def test_detection(args, cfg):
    _title("DETECTION TEST — OpenCV + YOLO fallback")
    from detection.detector import Detector
    det = Detector(cfg)
    n   = args.frames if not args.image else 1
    results = []

    for i in range(n):
        frame = _frame_from_args(args, cfg, "left")
        dec   = det.process(frame, state="TEST")
        results.append(dec)

        status = f"{'✓' if dec.cluster_found else '○'}  conf={dec.confidence:.2f}  [{dec.method}]  circles={len(dec.opencv_result.circles) if dec.opencv_result else 0}  {dec.latency_ms:.1f}ms"
        print(f"  Frame {i+1:>3}: {status}")

        if args.show:
            vis = frame.copy()
            # Draw all detected clusters in grey
            for i, (bx1,by1,bx2,by2) in enumerate(dec.all_boxes):
                sc = dec.all_scores[i] if i < len(dec.all_scores) else 0
                cv2.rectangle(vis,(bx1,by1),(bx2,by2),(160,160,160),2)
                cv2.putText(vis,f"{sc:.2f}",(bx1,by1-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,160,160),1)
            # Highlight selected POI in bright green
            if dec.cluster_box:
                x1,y1,x2,y2 = dec.cluster_box
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
                label = f"POI  conf={dec.confidence:.2f}  [{dec.method}]"
                cv2.putText(vis,label,(x1,max(y1-8,12)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            # Draw OpenCV circles if used
            if dec.opencv_result:
                for c in dec.opencv_result.circles:
                    cv2.circle(vis,(c.x,c.y),c.radius,(0,200,255),1)
            cv2.imshow("Detection Test", vis)
            if cv2.waitKey(0 if args.image else 30) & 0xFF == ord("q"): break

    cv2.destroyAllWindows()
    found  = sum(1 for r in results if r.cluster_found)
    avg_ms = sum(r.latency_ms for r in results) / len(results)
    _ok(f"Found: {found}/{len(results)}  avg={avg_ms:.1f}ms  ({1000/avg_ms:.1f} FPS)")
    if found == 0:
        _warn("No detections — run tools/tune_hsv.py to fix HSV thresholds.")
    return True


def test_seg(args, cfg):
    _title("SEGMENTATION TEST — USB cam + NCNN")
    from segmentation.segmentor import Segmentor
    from segmentation.cutting_point import extract_cutting_points
    seg = Segmentor(cfg)
    n   = args.frames if not args.image else 1

    for i in range(n):
        frame  = _frame_from_args(args, cfg, "usb")
        result = seg.segment(frame, state="TEST")
        cp     = extract_cutting_points(
            result.masks,
            method=cfg["segmentation"].get("cutting_point_method","topmost"),
            min_stem_area_px=cfg["segmentation"].get("min_stem_area_px",50),
        ) if result.success and result.masks is not None else None

        cuts = [(p.px, p.py) for p in cp.points] if cp else []
        print(f"  Frame {i+1:>3}: stems={result.stem_count}  "
              f"cuts={len(cuts)}  {result.latency_ms:.1f}ms  "
              f"{'✓' if result.success else '✗'}")

        if args.show:
            vis = frame.copy().astype(np.float32)
            if result.masks is not None:
                for m in result.masks:
                    vis[m] = vis[m]*0.4 + np.array([0,255,0],np.float32)*0.6
            vis = vis.astype(np.uint8)
            for cx,cy in cuts:
                cv2.drawMarker(vis,(cx,cy),(0,0,255),cv2.MARKER_CROSS,22,2)
            cv2.imshow("Seg Test", vis)
            if cv2.waitKey(0 if args.image else 30) & 0xFF == ord("q"): break

    cv2.destroyAllWindows()
    return True


def test_depth(args, cfg):
    _title("STEREO DEPTH TEST")
    from depth.stereo_depth import StereoDepth
    engine = StereoDepth(cfg)
    _ok(f"Calibrated: {engine._calibrated}  focal={engine.focal_px:.0f}px  B={engine.baseline_mm:.0f}mm")

    if args.image:
        frame = cv2.imread(args.image)
        h,w   = frame.shape[:2]
        dr    = engine.compute(frame, frame, (w//2, h//2), state="TEST")
        _warn(f"Same image both sides → disparity=0 expected. depth={dr.depth_mm}")
        return True

    from core.camera_manager import CameraManager
    cam = CameraManager(cfg); cam.open_all()
    for i in range(args.frames):
        ok, L, R = cam.read_stereo()
        if not ok: _warn(f"Frame {i+1}: stereo read failed."); continue
        h,w = L.shape[:2]
        dr  = engine.compute(L, R, (w//2, h//2), state="TEST")
        print(f"  Frame {i+1:>3}: depth={'%.1f'%dr.depth_mm if dr.depth_mm else 'N/A'}mm  cal={'Y' if dr.calibrated else 'N'}  {dr.latency_ms:.1f}ms")
        if args.show and dr.disparity_map is not None:
            vis = cv2.applyColorMap(
                cv2.normalize(dr.disparity_map,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U),
                cv2.COLORMAP_MAGMA)
            cv2.imshow("Disparity", vis)
            if cv2.waitKey(30) & 0xFF == ord("q"): break
    cam.close_all()
    cv2.destroyAllWindows()
    return True


def test_coords(args, cfg):
    _title("COORDINATE DEBUG TEST — Fake coords through pipeline")
    from logging_.coord_debug import CoordDebugLogger
    cdl = CoordDebugLogger(enabled=True, log_path="logs/test_coords.jsonl")

    # Simulate 5 POI + 5 CUT coordinate events
    for i in range(5):
        cdl.log("poi", x=320.0+i, y=240.0, z=450.0-i*10,
                conf=0.75, method="opencv", state="TEST")
        cdl.log("cut", x=200.0+i, y=180.0, z=440.0-i*10,
                stem_idx=0, area_px=350, state="TEST")

    recent = cdl.recent(10)
    _ok(f"Logged {len(recent)} events to logs/test_coords.jsonl")
    for e in recent:
        print(f"    [{e['ts_iso']}] {e['type'].upper()}  x={e['x']}  y={e['y']}  z={e['z']}")
    cdl.close()
    return True


def test_stm32(args, cfg):
    _title("STM32 COMMS TEST")
    if args.dry_run:
        cfg = dict(cfg)
        cfg["stm32"] = dict(cfg["stm32"])
        cfg["stm32"]["dry_run"] = True

    from logging_.coord_debug import CoordDebugLogger
    from comms.stm32_comms import STM32Comms
    cdl  = CoordDebugLogger(enabled=True, log_path="logs/test_stm32.jsonl")
    stm  = STM32Comms(cfg, cdl)
    ok   = stm.connect()

    if ok or args.dry_run:
        _ok("STM32 connected (or dry-run mode)")
        r = stm.move("forward", 50)
        print(f"  move forward: ok={r.ok}  {r.latency_ms:.1f}ms")
        r = stm.send_poi_coords(320, 240, 450, conf=0.8, method="opencv", state="TEST")
        print(f"  poi coords:   ok={r.ok}  {r.latency_ms:.1f}ms")
        r = stm.send_cut_coords(200, 180, 440, stem_idx=0, area_px=350, state="TEST")
        print(f"  cut coords:   ok={r.ok}  {r.latency_ms:.1f}ms")
        stm.stop()
        stm.close()
        cdl.close()
        _ok("STM32 test complete")
    else:
        _err("STM32 connection failed — check cable/WiFi and config.yaml")
    return ok or args.dry_run


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("subsystem",
                   choices=["cameras","detection","seg","depth","coords","stm32","all"])
    p.add_argument("--config",   default="config/config.yaml")
    p.add_argument("--image",    default=None)
    p.add_argument("--frames",   type=int, default=20)
    p.add_argument("--show",     action="store_true")
    p.add_argument("--dry-run",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")

    tests = {
        "cameras":   test_cameras,
        "detection": test_detection,
        "seg":       test_seg,
        "depth":     test_depth,
        "coords":    test_coords,
        "stm32":     test_stm32,
    }

    if args.subsystem == "all":
        results = []
        for name in ["cameras","detection","seg","depth","coords","stm32"]:
            ok = tests[name](args, cfg)
            results.append((name, ok))
        print(f"\n{'='*40}")
        print("  SUMMARY")
        print(f"{'='*40}")
        for name, ok in results:
            print(f"  {'✓' if ok else '✗'}  {name}")
        print()
    else:
        tests[args.subsystem](args, cfg)


if __name__ == "__main__":
    main()