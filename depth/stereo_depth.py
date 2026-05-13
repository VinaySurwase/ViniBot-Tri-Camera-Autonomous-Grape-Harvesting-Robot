"""
depth/stereo_depth.py — Stereo Depth Engine v2
===============================================
Computes depth at a pixel from a stereo pair of Pi Camera V3 frames.
Uses calibration from calibration/stereo_calib.npz when available.
Falls back to ORB feature matching when calibration is absent.

Z = (f * B) / disparity
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from logging_.logger import get_logger, log_event

log = get_logger("stereo_depth")


@dataclass
class DepthResult:
    depth_mm: Optional[float]
    disparity_map: Optional[np.ndarray]
    point_px: Optional[Tuple[int, int]]
    latency_ms: float = 0.0
    calibrated: bool  = False
    error: Optional[str] = None


class StereoDepth:
    def __init__(self, cfg: dict):
        d = cfg["depth"]
        self.baseline_mm  = d.get("baseline_mm", 60.0)
        self.focal_px     = d.get("focal_length_px", 700.0)
        self.max_depth    = d.get("max_depth_mm", 2000.0)
        self.min_depth    = d.get("min_depth_mm", 50.0)
        self.calib_file   = d.get("calibration_file", "calibration/stereo_calib.npz")
        self.enabled      = d.get("enabled", True)
        self.depth_window_px = int(d.get("depth_window_px", 21))
        self.min_valid_disp_px = int(d.get("min_valid_disp_px", 50))

        self._calibrated  = False
        self._map1L = self._map2L = self._map1R = self._map2R = self._Q = None

        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=512, blockSize=11,
            P1=8  * 3 * 121, P2=32 * 3 * 121,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        if self.enabled:
            self._load_calibration()

    def _load_calibration(self) -> None:
        p = Path(self.calib_file)
        if not p.exists():
            log.warning(f"Stereo calib not found: {p} — using fallback depth.")
            return
        try:
            data = np.load(str(p))
            self._map1L = data["map1L"]; self._map2L = data["map2L"]
            self._map1R = data["map1R"]; self._map2R = data["map2R"]
            self._Q     = data["Q"]
            if "focal_px"    in data: self.focal_px   = float(data["focal_px"])
            if "baseline_mm" in data: self.baseline_mm = float(data["baseline_mm"])
            self._calibrated = True
            log.info(f"Stereo calib loaded: f={self.focal_px:.1f}px  B={self.baseline_mm:.1f}mm")
        except Exception as e:
            log.error(f"Calib load error: {e}")

    def rectify_left(self, left: np.ndarray) -> np.ndarray:
        if not self._calibrated or self._map1L is None:
            return left
        return cv2.remap(left, self._map1L, self._map2L, cv2.INTER_LINEAR)
        
    def rectify_right(self, right: np.ndarray) -> np.ndarray:
        if not self._calibrated or self._map1R is None:
            return right
        return cv2.remap(right, self._map1R, self._map2R, cv2.INTER_LINEAR)

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
        point_px: Tuple[int, int],
        state: str = "",
    ) -> DepthResult:
        if not self.enabled:
            return DepthResult(depth_mm=None, disparity_map=None, point_px=point_px)

        t0 = time.perf_counter()
        try:
            if self._calibrated:
                depth_mm, disp = self._calibrated_depth(left, right, point_px)
            else:
                depth_mm, disp = self._orb_depth(left, right, point_px)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            log.error(f"Depth compute error: {e}")
            return DepthResult(depth_mm=None, disparity_map=None,
                               point_px=point_px, latency_ms=ms, error=str(e))

        ms = (time.perf_counter() - t0) * 1000
        if depth_mm is not None:
            depth_mm = float(np.clip(depth_mm, self.min_depth, self.max_depth))

        cal_str = "calibrated" if self._calibrated else "fallback"
        depth_str = f"{depth_mm:.1f}mm" if depth_mm is not None else "None"
        log.info(f"Depth @ {point_px}: {depth_str}  ({ms:.1f}ms)  [{cal_str}]")
        log_event(state, "depth_computed",
                  f"{depth_mm:.1f}" if depth_mm else "None",
                  latency_ms=ms,
                  extra=f"px={point_px},cal={self._calibrated}")

        return DepthResult(
            depth_mm=depth_mm, disparity_map=disp,
            point_px=point_px, latency_ms=ms, calibrated=self._calibrated,
        )

    def _calibrated_depth(
        self, left: np.ndarray, right: np.ndarray, pt: Tuple[int, int]
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        # Ensure odd window size for symmetric neighbourhood sampling
        if self.depth_window_px % 2 == 0:
            self.depth_window_px += 1

        lR = cv2.remap(left,  self._map1L, self._map2L, cv2.INTER_LINEAR)
        rR = cv2.remap(right, self._map1R, self._map2R, cv2.INTER_LINEAR)
        lG = cv2.cvtColor(lR, cv2.COLOR_BGR2GRAY)
        rG = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
        disp = self._stereo.compute(lG, rG).astype(np.float32) / 16.0
        x = int(np.clip(pt[0], 0, disp.shape[1] - 1))
        y = int(np.clip(pt[1], 0, disp.shape[0] - 1))
        d = disp[y, x]
        if d > 0:
            return (self.focal_px * self.baseline_mm) / d, disp

        # Fallback: center pixel often lands on low-texture/occluded regions.
        # Use a local robust median disparity from valid pixels to reduce jitter.
        half = self.depth_window_px // 2
        y1 = max(0, y - half)
        y2 = min(disp.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(disp.shape[1], x + half + 1)

        patch = disp[y1:y2, x1:x2]
        valid = patch[patch > 0]
        if valid.size < self.min_valid_disp_px:
            return None, disp

        # Keep central 90% disparities (robust to mismatches/outliers), then median.
        lo = np.percentile(valid, 5)
        hi = np.percentile(valid, 95)
        trimmed = valid[(valid >= lo) & (valid <= hi)]
        if trimmed.size == 0:
            return None, disp

        d_med = float(np.median(trimmed))
        if d_med <= 0:
            return None, disp
        return (self.focal_px * self.baseline_mm) / d_med, disp

    def _orb_depth(
        self, left: np.ndarray, right: np.ndarray, pt: Tuple[int, int]
    ) -> Tuple[Optional[float], None]:
        lG = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        rG = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(500)
        kp1, d1 = orb.detectAndCompute(lG, None)
        kp2, d2 = orb.detectAndCompute(rG, None)
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            return None, None
        matches = sorted(
            cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
            key=lambda m: m.distance
        )[:50]
        if not matches:
            return None, None
        best = min(matches,
                   key=lambda m: np.hypot(kp1[m.queryIdx].pt[0] - pt[0],
                                          kp1[m.queryIdx].pt[1] - pt[1]))
        disp = abs(kp1[best.queryIdx].pt[0] - kp2[best.trainIdx].pt[0])
        if disp < 0.5:
            return None, None
        return (self.focal_px * self.baseline_mm) / disp, None