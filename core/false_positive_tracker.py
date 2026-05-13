"""
core/false_positive_tracker.py — False Positive & Retry Manager
================================================================
Tracks:
  1. False-positive detection zones — areas where the detector fired
     but produced no valid seg result after max_retries.
     These zones are suppressed for a cooldown period.

  2. Per-stem retry counter — if segmentation on a given stem position
     fails min_stem_consistency times in a row it is marked as FP.

Both are logged and pushed to the Android debug panel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from logging_.logger import get_logger, log_event

log = get_logger("fp_tracker")


@dataclass
class FPZone:
    x1: int
    y1: int
    x2: int
    y2: int
    created_at: float = field(default_factory=time.time)
    cooldown_sec: float = 30.0
    reason: str = ""

    def expired(self) -> bool:
        return time.time() - self.created_at > self.cooldown_sec

    def overlaps(self, box: Tuple[int, int, int, int], iou_threshold: float = 0.3) -> bool:
        """Check if a detection box overlaps this FP zone above threshold."""
        bx1, by1, bx2, by2 = box
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return False
        area_b = (bx2 - bx1) * (by2 - by1)
        if area_b == 0:
            return False
        return (inter / area_b) >= iou_threshold


class FalsePositiveTracker:
    """
    Maintains a list of FP zones and per-stem retry counters.
    Call from the state machine after every detection / segmentation cycle.
    """

    def __init__(self, cfg: dict):
        d = cfg["detection"]
        s = cfg["segmentation"]
        self.max_det_retries    = d.get("max_detection_retries", 5)
        self.fp_iou_threshold   = d.get("fp_iou_threshold", 0.3)
        self.max_stem_retries   = s.get("max_stem_retries", 3)
        self.min_consistency    = s.get("min_stem_consistency", 2)

        self._fp_zones: List[FPZone] = []
        self._det_retry_count:  int  = 0
        self._stem_retry_count: int  = 0
        self._stem_seen_count:  int  = 0   # for consistency check

    # ── Detection FP ──────────────────────────────────────────────────────────

    def is_fp_zone(self, box: Tuple[int, int, int, int]) -> bool:
        """Return True if box overlaps an active FP zone (should skip this detection)."""
        self._purge_expired()
        for zone in self._fp_zones:
            if zone.overlaps(box, self.fp_iou_threshold):
                log.debug(f"Detection box {box} overlaps FP zone — skipping.")
                return True
        return False

    def record_det_failure(self, box: Tuple[int, int, int, int], state: str = "") -> bool:
        """
        Call when a detection cycle produced no valid seg result.
        Returns True when max retries exceeded and the zone should be blacklisted.
        """
        self._det_retry_count += 1
        log.debug(f"Det failure count: {self._det_retry_count}/{self.max_det_retries}")

        if self._det_retry_count >= self.max_det_retries:
            self._add_fp_zone(box, reason="max_det_retries")
            log_event(state=state, event="fp_zone_added",
                      value=str(box), extra=f"reason=max_det_retries")
            self._det_retry_count = 0
            return True
        return False

    def reset_det_retries(self) -> None:
        self._det_retry_count = 0

    def _add_fp_zone(self, box: Tuple[int, int, int, int], reason: str = "") -> None:
        x1, y1, x2, y2 = box
        zone = FPZone(x1=x1, y1=y1, x2=x2, y2=y2, reason=reason)
        self._fp_zones.append(zone)
        log.warning(f"FP zone added: {box}  reason={reason}  "
                    f"active zones: {len(self._fp_zones)}")

    def _purge_expired(self) -> None:
        before = len(self._fp_zones)
        self._fp_zones = [z for z in self._fp_zones if not z.expired()]
        purged = before - len(self._fp_zones)
        if purged:
            log.debug(f"Purged {purged} expired FP zones.")

    # ── Stem / segmentation FP ────────────────────────────────────────────────

    def record_stem_seen(self) -> bool:
        """
        Call each frame a stem is detected at the current position.
        Returns True when the stem has appeared enough times to be considered valid.
        """
        self._stem_seen_count += 1
        return self._stem_seen_count >= self.min_consistency

    def record_stem_failure(self, state: str = "") -> bool:
        """
        Call when segmentation / cutting point extraction failed for current stem.
        Returns True when max stem retries exceeded (treat as FP stem).
        """
        self._stem_retry_count += 1
        log.debug(f"Stem failure {self._stem_retry_count}/{self.max_stem_retries}")
        if self._stem_retry_count >= self.max_stem_retries:
            log.warning(f"Stem marked as false positive after {self._stem_retry_count} retries.")
            log_event(state=state, event="stem_fp",
                      extra=f"retries={self._stem_retry_count}")
            self.reset_stem()
            return True
        return False

    def reset_stem(self) -> None:
        self._stem_retry_count = 0
        self._stem_seen_count  = 0

    # ── Debug summary ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        self._purge_expired()
        return {
            "active_fp_zones":     len(self._fp_zones),
            "fp_zones":            [(z.x1, z.y1, z.x2, z.y2, z.reason) for z in self._fp_zones],
            "det_retry_count":     self._det_retry_count,
            "stem_retry_count":    self._stem_retry_count,
            "stem_seen_count":     self._stem_seen_count,
        }