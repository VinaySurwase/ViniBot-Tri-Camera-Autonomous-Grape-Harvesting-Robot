"""
segmentation/cutting_point.py — Cutting Point Extractor v2
===========================================================
Derives (x, y) pixel cutting coordinates from a binary stem mask.

Methods:
  topmost  — topmost pixel row (closest to where cutter reaches from above)
  centroid — geometric centre of mask
  midpoint — midpoint between topmost and centroid

The primary cutting point is chosen as the mask with the largest area
(most confident stem detection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from logging_.logger import get_logger, log_event

log = get_logger("cutting_point")


@dataclass
class CuttingPoint:
    px: int
    py: int
    stem_idx: int
    mask_area_px: int
    method: str
    depth_mm: Optional[float] = None


@dataclass
class CuttingPointResult:
    points: List[CuttingPoint] = field(default_factory=list)
    primary: Optional[CuttingPoint] = None


def extract_cutting_points(
    masks: np.ndarray,
    method: str = "topmost",
    min_stem_area_px: int = 50,
    state: str = "SEGMENT",
) -> CuttingPointResult:
    points: List[CuttingPoint] = []

    for idx, mask in enumerate(masks):
        area = int(mask.sum())
        if area < min_stem_area_px:
            log.debug(f"Stem {idx}: area {area}px < min {min_stem_area_px}px — skip")
            continue

        px, py = _point_from_mask(mask, method)
        points.append(CuttingPoint(
            px=px, py=py,
            stem_idx=idx,
            mask_area_px=area,
            method=method,
        ))
        log.debug(f"Stem {idx}: cut=({px},{py})  area={area}px  method={method}")

    if not points:
        log.warning("No valid cutting points extracted.")
        return CuttingPointResult()

    primary = max(points, key=lambda p: p.mask_area_px)
    log_event(state, "cutting_points_extracted", str(len(points)),
              extra=f"primary=({primary.px},{primary.py}),method={method}")

    return CuttingPointResult(points=points, primary=primary)


def _point_from_mask(mask: np.ndarray, method: str) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0

    if method == "topmost":
        top_y = int(ys.min())
        top_x = int(xs[ys == top_y].mean())
        return top_x, top_y

    elif method == "centroid":
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return int(xs.mean()), int(ys.mean())
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    elif method == "midpoint":
        top_y  = int(ys.min())
        top_x  = int(xs[ys == top_y].mean())
        M      = cv2.moments(mask.astype(np.uint8))
        cen_x  = int(M["m10"] / M["m00"]) if M["m00"] > 0 else int(xs.mean())
        cen_y  = int(M["m01"] / M["m00"]) if M["m00"] > 0 else int(ys.mean())
        return (top_x + cen_x) // 2, (top_y + cen_y) // 2

    else:
        log.warning(f"Unknown method '{method}' — using centroid fallback.")
        return int(xs.mean()), int(ys.mean())