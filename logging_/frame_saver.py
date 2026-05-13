"""
logging_/frame_saver.py — Annotated Frame Saver v2
===================================================
Saves annotated frames for offline visual review.
Burn-in overlays include: HUD bar, crosshairs, masks, bounding boxes,
cutting point marker, depth annotation, and FP zone markers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from logging_.logger import get_frames_dir, get_logger, log_event

log = get_logger("frame_saver")

_counters = {"det": 0, "seg": 0, "depth": 0}


def _dir() -> Optional[Path]:
    d = get_frames_dir()
    if d is None:
        log.warning("Frames dir not initialised.")
    return d


def save_detection_frame(
    frame: np.ndarray,
    circles: list,                                  # [(x,y,r), ...]
    cluster_box: Optional[Tuple[int, int, int, int]],
    confidence: float,
    state: str,
    method: str = "opencv",
    yolo_boxes: Optional[np.ndarray] = None,
    fp_zones: Optional[list] = None,               # [(x1,y1,x2,y2,reason)]
) -> Optional[str]:
    d = _dir()
    if d is None:
        return None

    img = frame.copy()

    # FP zones in red
    if fp_zones:
        for (zx1, zy1, zx2, zy2, reason) in fp_zones:
            cv2.rectangle(img, (zx1, zy1), (zx2, zy2), (0, 0, 200), 1)
            cv2.putText(img, f"FP:{reason[:8]}", (zx1, zy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    # HoughCircles
    for (cx, cy, r) in circles:
        cv2.circle(img, (cx, cy), r, (0, 200, 255), 1)
        cv2.circle(img, (cx, cy), 2, (0, 200, 255), -1)

    # YOLO fallback boxes in orange
    if yolo_boxes is not None:
        for box in yolo_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(img, "YOLO", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    # Confirmed cluster box in green
    if cluster_box:
        x1, y1, x2, y2 = cluster_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Cluster  conf={confidence:.2f}  [{method}]"
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    _hud(img, state, f"Det:{method}  circles:{len(circles)}")
    _counters["det"] += 1
    path = d / f"det_{_counters['det']:06d}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log_event(state, "frame_saved_det", str(path))
    return str(path)


def save_segmentation_frame(
    frame: np.ndarray,
    masks: Optional[np.ndarray],                   # [N, H, W] bool
    cutting_points: List[Tuple[int, int]],
    stem_count: int,
    state: str,
    depth_mm: Optional[float] = None,
) -> Optional[str]:
    d = _dir()
    if d is None:
        return None

    img = frame.copy().astype(np.float32)

    # Mask overlay (green)
    if masks is not None:
        color = np.array([0, 255, 0], dtype=np.float32)
        for mask in masks:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (img.shape[1], img.shape[0])).astype(bool)
            img[mask] = img[mask] * 0.4 + color * 0.6

    img = img.astype(np.uint8)

    # Cutting point markers
    for idx, (cx, cy) in enumerate(cutting_points):
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(img, (cx, cy), 8, (0, 0, 255), 2)
        lbl = f"CUT {idx+1}"
        if depth_mm is not None and idx == 0:
            lbl += f"  {depth_mm:.0f}mm"
        cv2.putText(img, lbl, (cx + 12, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    extra = f"stems:{stem_count}  cuts:{len(cutting_points)}"
    if depth_mm:
        extra += f"  depth:{depth_mm:.0f}mm"
    _hud(img, state, extra)

    _counters["seg"] += 1
    path = d / f"seg_{_counters['seg']:06d}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log_event(state, "frame_saved_seg", str(path),
              extra=f"stems={stem_count},depth={depth_mm}")
    return str(path)


def save_depth_frame(
    left: np.ndarray,
    right: np.ndarray,
    disparity: Optional[np.ndarray],
    cut_px: Optional[Tuple[int, int]],
    depth_mm: Optional[float],
    state: str,
) -> Optional[str]:
    d = _dir()
    if d is None:
        return None

    h, w  = left.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w]  = left
    canvas[:, w:]  = right

    if disparity is not None:
        vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
        vis = cv2.resize(vis, (w // 2, h // 2))
        canvas[h // 2:, :w // 2] = vis

    if cut_px:
        cv2.drawMarker(canvas, cut_px, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)

    label = f"Depth: {depth_mm:.1f}mm" if depth_mm else "Depth: N/A"
    cv2.putText(canvas, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, "LEFT",  (8,   h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "RIGHT", (w + 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    _hud(canvas, state, label)

    _counters["depth"] += 1
    path = d / f"depth_{_counters['depth']:06d}.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


def _hud(img: np.ndarray, state: str, extra: str = "") -> None:
    h, w = img.shape[:2]
    bar_h = 30
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    ts   = time.strftime("%H:%M:%S")
    text = f"{ts}  STATE:{state}"
    if extra:
        text += f"  |  {extra}"
    cv2.putText(img, text, (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)