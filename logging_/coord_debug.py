"""
logging_/coord_debug.py — Coordinate Debug Logger
==================================================
Writes every coordinate sent to the arm/STM32 to a JSONL file
so you can verify:
  • POI coordinates from stereo detection
  • Cutting point coordinates from USB segmentation
  • Whether depth values are sensible

Each line is a complete JSON object.

Enable via config: logging.coord_debug: true
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from logging_.logger import get_logger

log = get_logger("coord_debug")


class CoordDebugLogger:
    """
    Writes coordinate events to a JSONL file.
    Also keeps an in-memory ring buffer for the Android debug panel.
    """

    BUFFER_SIZE = 200

    def __init__(self, enabled: bool, log_path: str):
        self.enabled  = enabled
        self.log_path = Path(log_path)
        self._file    = None
        self._buffer: list[dict] = []

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "a")
            log.info(f"Coord debug logger active → {self.log_path}")
        else:
            log.info("Coord debug logger disabled.")

    def log(
        self,
        coord_type: str,        # "poi" | "cut"
        x: float,
        y: float,
        z: float,
        conf: float = 0.0,
        method: str = "",
        state: str = "",
        stem_idx: int = 0,
        area_px: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        if not self.enabled:
            return

        entry = {
            "ts":         time.time(),
            "ts_iso":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type":       coord_type,
            "x":          round(x, 3),
            "y":          round(y, 3),
            "z":          round(z, 3),
            "conf":       round(conf, 4),
            "method":     method,
            "state":      state,
            "stem_idx":   stem_idx,
            "area_px":    area_px,
        }
        if extra:
            entry.update(extra)

        # Write to file
        if self._file:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()

        # Ring buffer for Android debug panel
        self._buffer.append(entry)
        if len(self._buffer) > self.BUFFER_SIZE:
            self._buffer.pop(0)

    def recent(self, n: int = 20) -> list[dict]:
        """Return the last N coordinate entries."""
        return self._buffer[-n:]

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None