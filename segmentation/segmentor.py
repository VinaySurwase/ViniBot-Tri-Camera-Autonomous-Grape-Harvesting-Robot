"""
segmentation/segmentor.py — YOLOv8n-seg NCNN Segmentor v2
==========================================================
Receives BGR numpy frames from the USB camera.
Returns processed binary masks resized to frame dimensions.
Loaded once at init; never reloaded mid-run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from logging_.logger import get_logger, log_event

log = get_logger("segmentor")


@dataclass
class SegmentationResult:
    success: bool
    stem_count: int = 0
    masks: Optional[np.ndarray] = None     # [N, H, W] bool
    boxes: Optional[np.ndarray] = None     # [N, 4] xyxy
    scores: Optional[np.ndarray] = None    # [N] conf
    latency_ms: float = 0.0
    error: Optional[str] = None


class Segmentor:
    def __init__(self, cfg: dict):
        s = cfg["segmentation"]
        self.model_path     = s["model_path"]
        self.conf           = s.get("conf_thres", 0.15)
        self.imgsz          = s.get("imgsz", 480)
        self.mask_thr       = s.get("mask_threshold", 0.3)
        self.blur_k         = s.get("blur_kernel", 5)
        self.dilate_iters   = s.get("dilate_iterations", 1)
        self._dil_kernel    = np.ones((3, 3), np.uint8)
        self._model         = None

        if not Path(self.model_path).exists():
            log.error(f"Seg model not found: {self.model_path}")
        else:
            self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path, task="segment")
            log.info(f"Segmentation model loaded: {self.model_path}")
        except Exception as e:
            log.error(f"Seg model load failed: {e}")

    def segment(self, frame: np.ndarray, state: str = "SEGMENT") -> SegmentationResult:
        if self._model is None:
            return SegmentationResult(success=False, error="model not loaded")

        t0 = time.perf_counter()
        try:
            results = self._model(frame, imgsz=self.imgsz,
                                  conf=self.conf, verbose=False)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            log.error(f"Seg inference error: {e}")
            return SegmentationResult(success=False, latency_ms=ms, error=str(e))

        ms     = (time.perf_counter() - t0) * 1000
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            log.debug(f"No masks ({ms:.1f}ms)")
            return SegmentationResult(success=False, latency_ms=ms)

        raw_masks = result.masks.data.cpu().numpy()
        boxes     = result.boxes.xyxy.cpu().numpy()
        scores    = result.boxes.conf.cpu().numpy()
        H, W      = frame.shape[:2]

        processed = []
        for i, mask in enumerate(raw_masks):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.GaussianBlur(mask, (self.blur_k, self.blur_k), 0)
            m    = (mask > self.mask_thr).astype(np.uint8)
            m    = cv2.dilate(m, self._dil_kernel, iterations=self.dilate_iters)

            # Clip mask to its bounding box — prevents mask bleed outside
            # the detection region (raw YOLO masks are low-res and can spread)
            if i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                clip = np.zeros_like(m)
                clip[y1:y2, x1:x2] = m[y1:y2, x1:x2]
                m = clip

            processed.append(m.astype(bool))

        stem_count = len(processed)
        masks_arr  = np.stack(processed) if processed else None

        log.info(f"Segmentation: {stem_count} stem(s) ({ms:.1f}ms)")
        log_event(state, "segmentation", str(stem_count), latency_ms=ms,
                  extra=f"imgsz={self.imgsz},conf={self.conf}")

        return SegmentationResult(
            success=True,
            stem_count=stem_count,
            masks=masks_arr,
            boxes=boxes,
            scores=scores,
            latency_ms=ms,
        )