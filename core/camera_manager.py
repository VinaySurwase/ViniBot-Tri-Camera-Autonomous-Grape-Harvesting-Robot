"""
core/camera_manager.py — Tri-Camera Manager
============================================
Manages:
  • 2× Pi Camera V3   via Picamera2  (stereo detection)
  • 1× USB Camera     via OpenCV     (segmentation)

Cameras are opened independently. Stereo pair is captured synchronously
using Picamera2's synchronized capture where possible.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from logging_.logger import get_logger, log_event

log = get_logger("camera_manager")


# ── Pi Camera V3 wrapper ──────────────────────────────────────────────────────

class PiCameraV3:
    """Single Pi Camera V3 via Picamera2."""

    def __init__(self, index: int, width: int, height: int, fps: int,
                 warmup: int, role: str):
        self.index   = index
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.warmup  = warmup
        self.role    = role
        self._cam    = None
        self._lock   = threading.Lock()

    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
            self._cam = Picamera2(self.index)
            # Using create_preview_configuration instead of create_video_configuration
            # forces libcamera to select a full-FOV binned mode (like 2304x1296) rather
            # than a heavily cropped center window, maintaining a wide field of view.
            # Request exactly 2304x1296 (the native full FOV mode of IMX708). 
            # If we ask for 1280x720, libcamera might aggressively crop 1536x864 instead.
            # We will downscale to the requested size safely in the read() method.
            config = self._cam.create_preview_configuration(
                main={"size": (2304, 1296), "format": "RGB888"}
            )
            config["controls"] = {
                "FrameRate": float(self.fps),
                "AwbEnable": False,
                "ColourGains": (1.83, 1.60),   # calibrated to match rpicam-hello
            }
            self._cam.configure(config)
            self._cam.start()
            # Force full sensor crop to preserve maximum FOV.
            props = self._cam.camera_properties
            full_size = props.get("PixelArraySize", None)
            if full_size is not None and len(full_size) == 2:
                self._cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})
            # Warmup
            for _ in range(self.warmup):
                self._cam.capture_array()
            log.info(f"[{self.role}] Pi Camera V3 index={self.index} ready "
                     f"({self.width}×{self.height} @ {self.fps}fps)")
            log_event("INIT", "camera_open", self.role)
            return True
        except Exception as e:
            log.error(f"[{self.role}] Failed to open Pi Camera V3 index={self.index}: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cam is None:
            return False, None
        try:
            with self._lock:
                frame_rgb = self._cam.capture_array()
            # Picamera2 returns RGB; convert to BGR for OpenCV compatibility
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # Downscale from 2304x1296 to the requested size (e.g. 1280x720)
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
            return True, frame_bgr
        except Exception as e:
            log.warning(f"[{self.role}] Frame capture failed: {e}")
            return False, None

    def close(self) -> None:
        if self._cam:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass
            self._cam = None
            log.info(f"[{self.role}] Pi Camera closed.")


# ── USB Camera wrapper ────────────────────────────────────────────────────────

class USBCamera:
    """USB camera via OpenCV VideoCapture."""

    def __init__(self, index: Union[int, str], width: int, height: int, fps: int,
                 warmup: int, role: str):
        self.index  = index
        self.width  = width
        self.height = height
        self.fps    = fps
        self.warmup = warmup
        self.role   = role
        self._cap   = None

    def open(self) -> bool:
        try:
            idx = self.index
            if isinstance(idx, str) and idx.startswith('/dev/'):
                import os
                real_idx = os.path.realpath(idx)
                if real_idx.startswith('/dev/video'):
                    try:
                        idx = int(real_idx.replace('/dev/video', ''))
                    except ValueError:
                        pass
            self._cap = cv2.VideoCapture(idx)
            if not self._cap.isOpened():
                raise RuntimeError(f"VideoCapture({self.index}) not opened")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
            for _ in range(self.warmup):
                self._cap.read()
            log.info(f"[{self.role}] USB Camera index={self.index} ready "
                     f"({self.width}×{self.height} @ {self.fps}fps)")
            log_event("INIT", "camera_open", self.role)
            return True
        except Exception as e:
            log.error(f"[{self.role}] Failed to open USB camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            log.warning(f"[{self.role}] USB camera not open — attempting reconnect.")
            self.open()
        ok, frame = self._cap.read()
        if not ok:
            log.warning(f"[{self.role}] USB camera read failed.")
        return ok, frame if ok else None

    def close(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info(f"[{self.role}] USB camera released.")


# ── Tri-Camera Manager ────────────────────────────────────────────────────────

class CameraManager:
    """
    Manages all three cameras.

    Usage:
        cam = CameraManager(cfg)
        ok  = cam.open_all()

        # Detection (stereo)
        ok, left, right = cam.read_stereo()

        # Segmentation (USB)
        ok, seg_frame   = cam.read_seg()
    """

    def __init__(self, cfg: dict):
        cl = cfg["cameras"]

        sl = cl["stereo_left"]
        sr = cl["stereo_right"]
        us = cl["usb_seg"]

        self.left_cam = PiCameraV3(
            index=sl["index"], width=sl["width"], height=sl["height"],
            fps=sl["fps"], warmup=sl["warmup_frames"], role="STEREO_LEFT",
        )
        self.right_cam = PiCameraV3(
            index=sr["index"], width=sr["width"], height=sr["height"],
            fps=sr["fps"], warmup=sr["warmup_frames"], role="STEREO_RIGHT",
        )
        self.seg_cam = USBCamera(
            index=us["index"], width=us["width"], height=us["height"],
            fps=us["fps"], warmup=us["warmup_frames"], role="USB_SEG",
        )

    def open_all(self) -> bool:
        ok1 = self.left_cam.open()
        ok2 = self.right_cam.open()
        ok3 = self.seg_cam.open()
        if not all([ok1, ok2, ok3]):
            log.error(f"Camera init: left={ok1} right={ok2} seg={ok3}")
        return all([ok1, ok2, ok3])

    def read_stereo(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame from each stereo camera as close in time as possible."""
        ok1, left  = self.left_cam.read()
        ok2, right = self.right_cam.read()
        return (ok1 and ok2), left, right

    def read_seg(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.seg_cam.read()

    def close_all(self) -> None:
        self.left_cam.close()
        self.right_cam.close()
        self.seg_cam.close()