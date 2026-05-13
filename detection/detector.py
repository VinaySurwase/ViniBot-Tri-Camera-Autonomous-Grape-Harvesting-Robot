"""
detection/detector.py — Detection Orchestrator v2
==================================================
Primary: YOLOv8n NCNN on left stereo frame
Fallback: OpenCV HSV + HoughCircles if YOLO finds nothing

Input: BGR numpy array (from PiCameraV3, already converted from RGB)
Output: DetectionDecision dataclass

Confirmation window: cluster must appear in N consecutive frames
before the state machine is allowed to transition to APPROACH.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from logging_.logger import get_logger, log_event

log = get_logger("detector")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Circle:
    x: int
    y: int
    radius: int


@dataclass
class Cluster:
    box: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    centre: Tuple[int, int]
    circle_count: int
    confidence: float


@dataclass
class OpenCVResult:
    success: bool
    circles: List[Circle] = field(default_factory=list)
    best_cluster: Optional[Cluster] = None
    debug_mask: Optional[np.ndarray] = None
    latency_ms: float = 0.0


@dataclass
class YOLOResult:
    success: bool
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    best_box: Optional[Tuple[int, int, int, int]] = None   # largest-area cluster (POI)
    best_score: float = 0.0
    latency_ms: float = 0.0


@dataclass
class DetectionDecision:
    cluster_found: bool
    cluster_box: Optional[Tuple[int, int, int, int]]        # chosen POI box
    cluster_centre: Optional[Tuple[int, int]]
    confidence: float
    method: str                         # "opencv" | "yolo" | "none"
    confirmed: bool                     # True after N consecutive positives
    all_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)   # every detected cluster
    all_scores: List[float] = field(default_factory=list)
    opencv_result: Optional[OpenCVResult] = None
    yolo_result: Optional[YOLOResult] = None
    latency_ms: float = 0.0


# ── OpenCV detector ───────────────────────────────────────────────────────────

class OpenCVDetector:
    def __init__(self, cfg: dict):
        d = cfg["detection"]
        self.hsv_lower     = np.array(d["hsv_lower"], dtype=np.uint8)
        self.hsv_upper     = np.array(d["hsv_upper"], dtype=np.uint8)
        self.morph_k       = d.get("morph_kernel_size", 5)
        self.min_area      = d.get("min_contour_area", 300)
        self.hough_dp      = d.get("hough_dp", 1.2)
        self.hough_dist    = d.get("hough_min_dist", 30)
        self.hough_p1      = d.get("hough_param1", 50)
        self.hough_p2      = d.get("hough_param2", 30)
        self.min_r         = d.get("min_circle_radius", 8)
        self.max_r         = d.get("max_circle_radius", 60)
        self.cluster_eps   = d.get("cluster_eps", 80)
        self.min_circles   = d.get("min_circles_for_cluster", 2)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_k, self.morph_k))

    def detect(self, frame: np.ndarray) -> OpenCVResult:
        t0 = time.perf_counter()

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Filter tiny contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean   = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) >= self.min_area:
                cv2.drawContours(clean, [c], -1, 255, -1)

        ms = (time.perf_counter() - t0) * 1000

        if clean.sum() == 0:
            return OpenCVResult(success=False, latency_ms=ms, debug_mask=clean)

        grey = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), clean)
        grey = cv2.GaussianBlur(grey, (9, 9), 2)

        raw = cv2.HoughCircles(
            grey, cv2.HOUGH_GRADIENT,
            dp=self.hough_dp, minDist=self.hough_dist,
            param1=self.hough_p1, param2=self.hough_p2,
            minRadius=self.min_r, maxRadius=self.max_r,
        )

        circles: List[Circle] = []
        if raw is not None:
            for x, y, r in np.round(raw[0]).astype(int):
                circles.append(Circle(int(x), int(y), int(r)))

        ms = (time.perf_counter() - t0) * 1000
        if not circles:
            return OpenCVResult(success=False, circles=[], latency_ms=ms, debug_mask=clean)

        best = self._best_cluster(circles)
        return OpenCVResult(
            success=best is not None,
            circles=circles,
            best_cluster=best,
            latency_ms=ms,
            debug_mask=clean,
        )

    def _best_cluster(self, circles: List[Circle]) -> Optional[Cluster]:
        if not circles:
            return None
        used  = [False] * len(circles)
        best: Optional[Cluster] = None

        for i, ci in enumerate(circles):
            if used[i]:
                continue
            group = [i]; used[i] = True
            for j, cj in enumerate(circles):
                if not used[j] and np.hypot(ci.x - cj.x, ci.y - cj.y) <= self.cluster_eps:
                    group.append(j); used[j] = True

            if len(group) < self.min_circles:
                continue

            xs = [circles[k].x for k in group]
            ys = [circles[k].y for k in group]
            rs = [circles[k].radius for k in group]
            x1, y1 = max(0, min(xs) - max(rs)), max(0, min(ys) - max(rs))
            x2, y2 = max(xs) + max(rs), max(ys) + max(rs)
            conf   = min(1.0, len(group) / 10.0 + 0.3)
            cl     = Cluster(
                box=(x1, y1, x2, y2),
                centre=((x1 + x2) // 2, (y1 + y2) // 2),
                circle_count=len(group),
                confidence=conf,
            )
            if best is None or cl.circle_count > best.circle_count:
                best = cl

        return best


# ── YOLO primary ──────────────────────────────────────────────────────────────

class YOLODetector:
    def __init__(self, cfg: dict):
        import platform
        d = cfg["detection"]
        ncnn_path  = d.get("yolo_model_path", "weights/det_ncnn_model")
        pt_path    = d.get("yolo_pt_model_path", "weights/det/best.pt")
        auto       = d.get("auto_select_format", True)
        self.conf  = d.get("yolo_conf_thres", 0.25)
        self.iou   = d.get("yolo_iou_thres", 0.45)
        self._model = None

        # Always use NCNN as requested by matching grape_detect_ncnn.py
        use_pt = False

        if use_pt:
            self.model_path = pt_path
            self.imgsz      = 640          # .pt handles any size natively
            self._format    = "pt"
        else:
            self.model_path = ncnn_path
            self.imgsz      = d.get("yolo_imgsz", 640)  # must match NCNN export
            self._format    = "ncnn"

        if Path(self.model_path).exists():
            self._load()
        else:
            log.warning(f"YOLO model not found: {self.model_path}")

    def _load(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path, task="detect")
            log.info(f"YOLO loaded [{self._format}]: {self.model_path}")
        except Exception as e:
            log.error(f"YOLO load failed: {e}")

    def detect(self, frame: np.ndarray) -> YOLOResult:
        if self._model is None:
            return YOLOResult(success=False)
        t0 = time.perf_counter()
        try:
            res = self._model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
            
            ms = (time.perf_counter() - t0) * 1000
            
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confidences = res.boxes.conf.cpu().numpy()
            
            boxes = []
            scores = []
            for box, sc in zip(boxes_xyxy, confidences):
                boxes.append(tuple(map(int, box)))
                scores.append(float(sc))
                
            if not boxes:
                return YOLOResult(success=False, latency_ms=ms)
                
            # Select the detection with the highest confidence score
            best_i = int(np.argmax(scores))
            
            return YOLOResult(
                success=True, boxes=boxes, scores=scores,
                best_box=boxes[best_i], best_score=scores[best_i],
                latency_ms=ms,
            )
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            log.error(f"YOLO inference error: {e}")
            return YOLOResult(success=False, latency_ms=ms)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Detector:
    """
    Runs YOLO → OpenCV fallback.
    Tracks N-frame confirmation window before declaring a confirmed POI.
    """

    def __init__(self, cfg: dict):
        d = cfg["detection"]
        self._yolo          = YOLODetector(cfg)
        self._opencv        = OpenCVDetector(cfg) if d.get("fallback_to_opencv", True) else None
        self._min_conf      = d.get("min_cluster_confidence", 0.55)
        self._confirm_n     = d.get("frames_to_confirm", 3)
        self._recent: deque = deque(maxlen=self._confirm_n)

    def process(self, frame: np.ndarray, state: str = "DETECT") -> DetectionDecision:
        t0 = time.perf_counter()

        cluster_found  = False
        cluster_box    = None
        cluster_centre = None
        confidence     = 0.0
        method         = "none"
        ocv            = None
        yolo_res       = None
        all_boxes: List[Tuple[int, int, int, int]] = []
        all_scores: List[float] = []

        # ── YOLO primary ──────────────────────────────────────────────────────
        yolo_res = self._yolo.detect(frame)
        if yolo_res.success and yolo_res.best_box:
            x1, y1, x2, y2 = yolo_res.best_box
            cluster_found  = True
            cluster_box    = yolo_res.best_box
            cluster_centre = ((x1 + x2) // 2, (y1 + y2) // 2)
            confidence     = yolo_res.best_score
            method         = "yolo"
            all_boxes      = yolo_res.boxes
            all_scores     = yolo_res.scores

        # ── OpenCV fallback ───────────────────────────────────────────────────
        elif self._opencv is not None:
            ocv = self._opencv.detect(frame)
            if ocv.success and ocv.best_cluster:
                bc             = ocv.best_cluster
                cluster_found  = True
                cluster_box    = bc.box
                cluster_centre = bc.centre
                confidence     = bc.confidence
                method         = "opencv"

        # ── Confidence gate ───────────────────────────────────────────────────
        if cluster_found and confidence < self._min_conf:
            cluster_found = False; method = "none"

        # ── Confirmation window ───────────────────────────────────────────────
        self._recent.append(cluster_found)
        confirmed = (
            len(self._recent) == self._confirm_n and all(self._recent)
        )

        ms = (time.perf_counter() - t0) * 1000

        if confirmed:
            log_event(state, "cluster_confirmed", method,
                      latency_ms=ms,
                      extra=f"conf={confidence:.2f},box={cluster_box}")

        return DetectionDecision(
            cluster_found=cluster_found,
            cluster_box=cluster_box,
            cluster_centre=cluster_centre,
            confidence=confidence,
            method=method,
            confirmed=confirmed,
            all_boxes=all_boxes,
            all_scores=all_scores,
            opencv_result=ocv,
            yolo_result=yolo_res,
            latency_ms=ms,
        )

    def reset_confirmation(self) -> None:
        self._recent.clear()