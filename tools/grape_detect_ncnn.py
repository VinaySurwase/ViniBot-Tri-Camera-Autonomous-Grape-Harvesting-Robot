#!/usr/bin/env python3
"""
Grape Detection — YOLOv8n NCNN Inference
=========================================
Project layout:

    root/
    ├── photo.jpg                      ← images sit directly here
    ├── another.jpg
    ├── weights/
    │   └── det_ncnn_model/
    │       ├── model.ncnn.bin
    │       ├── model.ncnn.param
    │       ├── model_ncnn.py
    │       └── metadata.yaml
    └── tools/
        └── grape_detect_ncnn.py       ← this file

Usage (run from root OR from tools/):
    python tools/grape_detect_ncnn.py --image photo.jpg
    python tools/grape_detect_ncnn.py --image photo.jpg --conf 0.40 --iou 0.30 --save
    python tools/grape_detect_ncnn.py --image photo.jpg --no-show --save

Requirements:
    pip install ultralytics opencv-python numpy
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── Resolve root dir relative to this script ──────────────────────────────
# tools/grape_detect_ncnn.py → .parent = tools/ → .parent = root/
ROOT_DIR      = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT_DIR / "weights" / "det_ncnn_model"
# Images live directly in root/ (no subfolder)

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES  = ["grapes"]
INPUT_SIZE   = 640          # must match the size used during ncnn export
DEFAULT_CONF = 0.25
DEFAULT_IOU  = 0.45
COLOR        = (0, 200, 80)  # BGR green


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  ultralytics not installed.  Run: pip install ultralytics")

    param_file = model_dir / "model.ncnn.param"
    bin_file   = model_dir / "model.ncnn.bin"

    if not param_file.exists() or not bin_file.exists():
        sys.exit(
            f"❌  NCNN model files not found in:\n"
            f"      {model_dir}\n"
            f"    Expected: model.ncnn.param  and  model.ncnn.bin"
        )

    print(f"📦  Loading NCNN model from : {model_dir}")
    model = YOLO(str(model_dir), task="detect")
    print("✅  Model loaded.")
    return model


# ── Inference ──────────────────────────────────────────────────────────────

def run_inference(model, image_path: Path, conf: float, iou: float):
    if not image_path.exists():
        sys.exit(f"❌  Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        sys.exit(f"❌  Could not read image: {image_path}")

    print(f"🔍  Image : {image_path.name}  ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)")
    print(f"    conf={conf}  iou={iou}")

    t0 = time.perf_counter()
    results = model.predict(
        source  = str(image_path),
        imgsz   = INPUT_SIZE,
        conf    = conf,
        iou     = iou,
        verbose = False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result      = results[0]
    boxes_xyxy  = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids   = result.boxes.cls.cpu().numpy().astype(int)

    print(f"⏱️   Inference : {elapsed_ms:.1f} ms")
    print(f"🍇  Detections: {len(boxes_xyxy)}")

    # ── Draw detections ────────────────────────────────────────────────────
    annotated = img_bgr.copy()
    for (x1, y1, x2, y2), conf_val, cls_id in zip(boxes_xyxy, confidences, class_ids):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{conf_val:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 4, th + baseline)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - baseline),
            (x1 + tw, label_y + baseline),
            COLOR, cv2.FILLED,
        )
        cv2.putText(
            annotated, label,
            (x1, label_y - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )
        print(f"   [{cls_id}] {CLASS_NAMES[cls_id]}  conf={conf_val:.3f}  "
              f"box=({x1},{y1},{x2},{y2})")

    # ── Detection count — bottom-left corner, away from model labels ───────
    h_img = annotated.shape[0]
    banner = f"Grapes detected: {len(boxes_xyxy)}"
    cv2.putText(annotated, banner, (10, h_img - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR, 2, cv2.LINE_AA)

    return annotated, results


# ── Display ────────────────────────────────────────────────────────────────

def show_image(annotated_bgr, title: str = "Grape Detection — press any key to close"):
    h, w = annotated_bgr.shape[:2]
    max_disp = 1280
    if max(h, w) > max_disp:
        scale = max_disp / max(h, w)
        disp  = cv2.resize(annotated_bgr, (int(w * scale), int(h * scale)))
    else:
        disp = annotated_bgr

    cv2.imshow(title, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Save ───────────────────────────────────────────────────────────────────

def save_image(annotated_bgr, source_path: Path) -> Path:
    out = source_path.parent / f"{source_path.stem}_detected{source_path.suffix}"
    cv2.imwrite(str(out), annotated_bgr)
    print(f"💾  Saved → {out}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8n NCNN Grape Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", required=True,
        help="Filename or path of input image. Bare filenames are looked up in root/ "
             "(e.g. --image photo.jpg  →  root/photo.jpg)"
    )
    p.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL),
        help="Path to NCNN model folder (contains model.ncnn.param + model.ncnn.bin)"
    )
    p.add_argument("--conf",    type=float, default=DEFAULT_CONF,
                   help="Confidence threshold (0–1)")
    p.add_argument("--iou",     type=float, default=DEFAULT_IOU,
                   help="NMS IoU threshold (0–1)")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated image alongside the source image")
    p.add_argument("--no-show", action="store_true",
                   help="Skip display window (useful on headless / SSH sessions)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve image: if bare filename or relative path doesn't exist as-is,
    # look it up directly in root/
    img_path = Path(args.image)
    if not img_path.is_absolute() and not img_path.exists():
        img_path = ROOT_DIR / img_path

    model_dir = Path(args.model_dir)

    model        = load_model(model_dir)
    annotated, _ = run_inference(model, img_path, args.conf, args.iou)

    if args.save:
        save_image(annotated, img_path)

    if not args.no_show:
        show_image(annotated)


if __name__ == "__main__":
    main()