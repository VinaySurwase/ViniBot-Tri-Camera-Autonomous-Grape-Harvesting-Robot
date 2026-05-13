"""
core/state_machine.py — GrapeBot v2 State Machine
===================================================

States:
  IDLE       → waiting for Android start or operator
  DETECT     → stereo V3 cams active; OpenCV+YOLO detection for grape cluster POI
  APPROACH   → POI confirmed; depth computed; coords sent to STM32/arm; bot moves toward cluster
  SEGMENT    → bot at position; USB cam runs seg model; cutting point found
  CUT        → cut coords sent to STM32 arm; arm executes cut
  MANUAL     → direct Android joystick control; all auto logic paused
  ERROR      → unrecoverable failure; wait for reset

Chassis control: ESP32 (over UART2) — 4 Johnson motors via BTS7960B
Arm/Cutter:      STM32 (over UART0) — unchanged

Android control:
  Any state → MANUAL if Android sends mode=manual
  MANUAL    → DETECT if Android sends mode=auto + cmd=start
  Android move commands forwarded to ESP32 in MANUAL mode
"""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import cv2

from comms.stm32_comms import STM32Comms
from comms.esp32_comms import ESP32Comms
from comms.ws_server import WSServer
from core.camera_manager import CameraManager
from core.false_positive_tracker import FalsePositiveTracker
from depth.stereo_depth import StereoDepth
from detection.detector import Detector
from logging_.coord_debug import CoordDebugLogger
from logging_.frame_saver import save_detection_frame, save_segmentation_frame, save_depth_frame
from logging_.logger import get_logger, log_event
from segmentation.cutting_point import extract_cutting_points
from segmentation.segmentor import Segmentor

log = get_logger("state_machine")


class State(Enum):
    IDLE     = auto()
    DETECT   = auto()
    APPROACH = auto()
    SEGMENT  = auto()
    CUT      = auto()
    MANUAL   = auto()
    ERROR    = auto()


class GrapeBotFSM:
    """
    Central state machine. Runs in its own thread after start().
    Android commands arrive via WSServer callbacks (from WS thread).
    All state mutations are protected by _state_lock.
    """

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self._state     = State.IDLE
        self._mode      = "manual"          # "auto" | "manual"
        self._running   = False
        self._state_lock = threading.Lock()
        self._state_data: dict = {}

        sm = cfg["state_machine"]
        self._detect_hz = sm.get("detect_loop_hz", 15)
        self._seg_hz    = sm.get("seg_loop_hz", 10)

        lg = cfg["logging"]
        self._save_frames    = lg.get("save_frames", True)
        self._det_interval   = lg.get("det_frame_interval", 5)
        self._seg_save_every = lg.get("seg_save_every", True)
        self._frame_tick     = 0

        # Sub-systems
        self.cameras   = CameraManager(cfg)
        self.detector  = Detector(cfg)
        self.seg       = Segmentor(cfg)
        self.depth     = StereoDepth(cfg)

        self.coord_debug = CoordDebugLogger(
            enabled=lg.get("coord_debug", True),
            log_path=lg.get("coord_log_file", "logs/coords.jsonl"),
        )
        self.stm32  = STM32Comms(cfg, self.coord_debug)   # arm + cutter only
        self.esp32  = ESP32Comms(cfg)                      # chassis movement
        self.ws     = WSServer(cfg)
        self.fp     = FalsePositiveTracker(cfg)

        # Register WS callbacks
        self.ws.on_mode_change(self._android_mode_change)
        self.ws.on_move_cmd(self._android_move)
        self.ws.on_control_cmd(self._android_control)

        # Runtime data shared across ticks
        self._poi_box:    Optional[tuple] = None
        self._poi_centre: Optional[tuple] = None
        self._poi_conf:   float = 0.0
        self._poi_method: str   = ""
        self._poi_depth:  Optional[float] = None

        # Chassis control (RPi-driven, no ESP32 autoLoop needed)
        self._chassis_driving: bool  = False
        self._drive_speed:    int    = sm.get("drive_speed", 50)

        # Visual servoing state (used in SEGMENT)
        self._servo_angle:      float = 22.0
        self._last_servo_send:  float = 0.0
        self._last_sent_angle:  int   = -1
        self._servo_start_time: float = 0.0   # when first servo was sent
        self._servo_wait_sec:   float = sm.get("servo_settle_sec", 3.0)

        # Scissor cutting state (used in CUT)
        self._cut_count: int = 0

        # Color-based fallback detector
        self._color_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._color_confirm_count: int = 0

        log.info("GrapeBotFSM v2 initialised.")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        log.info("Opening cameras…")
        if not self.cameras.open_all():
            log.error("Camera init failed.")
            self._transition(State.ERROR, reason="camera_init_failed")

        self.stm32.connect()     # arm controller
        self.esp32.connect()     # chassis controller
        self.ws.start()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=False, name="fsm-main")
        self._thread.start()
        log.info("State machine running.")

    def stop(self) -> None:
        self._running = False
        self.esp32.stop()       # halt chassis immediately
        self.stm32.stop()       # safe-home the arm
        self.ws.stop()
        self.cameras.close_all()
        self.coord_debug.close()
        from logging_.logger import close_logs
        close_logs()
        log.info("GrapeBot shutdown.")

    def _loop(self) -> None:
        while self._running:
            with self._state_lock:
                state = self._state
            try:
                {
                    State.IDLE:     self._tick_idle,
                    State.DETECT:   self._tick_detect,
                    State.APPROACH: self._tick_approach,
                    State.SEGMENT:  self._tick_segment,
                    State.CUT:      self._tick_cut,
                    State.MANUAL:   self._tick_manual,
                    State.ERROR:    self._tick_error,
                }[state]()
            except Exception as exc:
                log.exception(f"Unhandled exception in state {state.name}: {exc}")
                self._transition(State.ERROR, reason=str(exc))

    # =========================================================================
    # Android callbacks (called from WS thread — thread safe)
    # =========================================================================

    def _android_mode_change(self, mode: str) -> None:
        log.info(f"Android mode → {mode}")
        self._mode = mode
        if mode == "manual":
            self._transition(State.MANUAL, reason="android_manual")
        elif mode == "auto":
            if self._state == State.MANUAL:
                self._transition(State.IDLE, reason="android_auto")

    def _android_move(self, direction: str, speed: int) -> None:
        if self._state != State.MANUAL:
            log.debug(f"Move command ignored — not in MANUAL state (state={self._state.name})")
            return
        log.debug(f"Manual move → ESP32: {direction} speed={speed}")
        resp = self.esp32.move(direction, speed)
        if not resp.ok:
            log.warning(f"ESP32 move failed: {resp.error}")

    def _android_control(self, cmd: str) -> None:
        log.info(f"Android control cmd: {cmd}")
        if cmd == "start" and self._mode == "auto":
            # Keep ESP32 in MANUAL mode — RPi controls chassis directly
            self.esp32.set_mode("manual")
            self._chassis_driving = False
            self._transition(State.DETECT, reason="android_start")
        elif cmd == "stop":
            self.esp32.stop()            # halt chassis
            self._chassis_driving = False
            self.stm32.stop()            # safe-stop arm
            self._transition(State.IDLE, reason="android_stop")
        elif cmd == "home":
            self.stm32.home()            # arm only
        elif cmd == "reset":
            self.fp.reset_det_retries()
            self.fp.reset_stem()
            self._chassis_driving = False
            self._transition(State.IDLE, reason="android_reset")

    # =========================================================================
    # State tick handlers
    # =========================================================================

    def _tick_idle(self) -> None:
        self.ws.push_state("IDLE", self._mode)
        time.sleep(0.2)

    def _tick_detect(self) -> None:
        """
        DETECT — stereo V3 cameras only.
        Uses YOLO first; if YOLO fails, uses color-based fallback.
        """
        t0 = time.time()
        ok, left, right = self.cameras.read_stereo()
        if not ok:
            log.warning("Stereo read failed in DETECT.")
            time.sleep(0.1)
            return

        # ── Primary: YOLO detection ───────────────────────────────────────
        decision = self.detector.process(left, state="DETECT")
        self._frame_tick += 1
        detection_method = "yolo"

        # ── Fallback: Color-based detection ───────────────────────────────
        if not decision.cluster_found:
            color_result = self._color_detect(left)
            if color_result is not None:
                box, centre = color_result
                # Patch the decision object so the rest of the pipeline works
                decision.cluster_found = True
                decision.cluster_box = box
                decision.cluster_centre = centre
                decision.confirmed = (self._color_confirm_count >= 3)
                decision.confidence = 0.8
                decision.method = "color"
                detection_method = "color"

        # Push live detection data to Android
        circles = 0
        if hasattr(decision, 'opencv_result') and decision.opencv_result:
            circles = len(decision.opencv_result.circles)
        self.ws.push_detection(
            found=decision.cluster_found,
            cluster_box=decision.cluster_box,
            conf=getattr(decision, 'confidence', 0.0),
            method=detection_method,
            circles=circles,
        )

        # Save annotated frame
        if self._save_frames and self._frame_tick % self._det_interval == 0:
            circle_list = []
            if hasattr(decision, 'opencv_result') and decision.opencv_result and decision.opencv_result.circles:
                circle_list = [(c.x, c.y, c.radius) for c in decision.opencv_result.circles]
            yolo_boxes = None
            if hasattr(decision, 'yolo_result') and decision.yolo_result and decision.yolo_result.boxes:
                yolo_boxes = np.array([list(b) for b in decision.yolo_result.boxes])
            save_detection_frame(
                frame=left,
                circles=circle_list,
                cluster_box=decision.cluster_box,
                confidence=getattr(decision, 'confidence', 0.0),
                state="DETECT",
                method=detection_method,
                yolo_boxes=yolo_boxes,
            )

        # FP zone check
        if decision.cluster_box and self.fp.is_fp_zone(decision.cluster_box):
            log.debug("Cluster box in FP zone — ignoring.")
            time.sleep(1.0 / self._detect_hz)
            return

        if decision.confirmed and decision.cluster_box:
            self._poi_box    = decision.cluster_box
            self._poi_centre = decision.cluster_centre
            self._poi_conf   = getattr(decision, 'confidence', 0.8)
            self._poi_method = detection_method

            # Compute depth before approach
            dr = self.depth.compute(left, right, self._poi_centre, state="DETECT")
            self._poi_depth = dr.depth_mm

            depth_str = f"{self._poi_depth:.1f}mm" if self._poi_depth else "N/A"
            log.info(f"POI confirmed ({detection_method}): centre={self._poi_centre}  "
                     f"depth={depth_str}  conf={self._poi_conf:.2f}")

            # STOP the chassis so the bot holds position for APPROACH/SEGMENT
            self.esp32.stop()
            self._chassis_driving = False
            log.info("Target detected — chassis stopped (ESP32).")

            self.fp.reset_det_retries()
            self._color_confirm_count = 0
            self._transition(State.APPROACH, reason=f"poi_confirmed_{detection_method}")

        else:
            # No target found — drive forward (RPi-controlled)
            if not self._chassis_driving:
                log.info("No target detected — driving forward.")
                self.esp32.move("forward", self._drive_speed)
                self._chassis_driving = True

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0 / self._detect_hz - elapsed))

    def _tick_approach(self) -> None:
        """
        APPROACH — send POI coords to STM32; bot navigates to cluster.
        No camera inference runs here.
        """
        if not self._poi_centre:
            log.error("No POI centre in APPROACH — back to DETECT.")
            self._transition(State.DETECT, reason="no_poi")
            return

        cx, cy = self._poi_centre
        depth_mm = self._poi_depth or 0.0

        # ── Depth safety constraint ────────────────────────────────────────────
        # Only send coords if depth is valid and within arm reach range.
        if not (150 < depth_mm <= 650):
            log.warning(f"Depth out of range ({depth_mm:.1f}mm) — need 150–650mm. Back to DETECT.")
            self._transition(State.DETECT, reason="depth_out_of_range")
            return

        W = self.cfg["cameras"]["stereo_left"]["width"]
        H = self.cfg["cameras"]["stereo_left"]["height"]

        send_x = 90.0 - ((cx / W) * 180.0)
        send_y = depth_mm

        # ── Z: Height in arm coordinate system ─────────────────────────────────
        # Reference (0) is 390mm below camera centre.
        # Arm travels 0 → 325mm. Cannot go below 0.
        # Pinhole model: vertical offset = depth * (H/2 - cy) / focal
        #   cy < H/2 → grape above camera centre → higher Z
        #   cy > H/2 → grape below camera centre → lower Z
        CAM_HEIGHT_FROM_REF = 390.0
        ARM_MAX_MM          = 325.0
        focal = self.depth.focal_px if self.depth.enabled else (H * 0.8)
        vert_offset = depth_mm * (H / 2.0 - cy) / focal if depth_mm > 0 else 0.0
        send_z = CAM_HEIGHT_FROM_REF + vert_offset - 115.0
        send_z = max(0.0, min(send_z, ARM_MAX_MM))

        log.info(f"Sending POI coords to STM32: X={send_x:.1f} Y={send_y:.1f} Z={send_z:.1f}")
        resp = self.stm32.send_poi_coords(
            x=send_x, y=send_y, z=send_z,
            conf=self._poi_conf,
            method=self._poi_method,
            state="APPROACH",
        )
        self.ws.push_coords_poi(send_x, send_y, send_z, self._poi_conf, self._poi_method)

        if not resp.ok:
            log.warning(f"STM32 POI send failed: {resp.error}")
            # Retry logic handled externally; transition to SEGMENT anyway
            # (arm may still be moving)

        # Wait for STM32 to signal arrival (or timeout)
        log.info("Waiting for bot to reach cluster position…")
        arrived = self._wait_stm32_arrive(timeout_sec=10.0)
        if not arrived:
            log.warning("Approach timeout — proceeding anyway (arm may be close enough).")

        self.fp.reset_stem()
        self._transition(State.SEGMENT, reason="approach_done")

    def _tick_segment(self) -> None:
        """
        SEGMENT — Visual servoing with USB cam.
        Adjusts end-effector servo to align cutting point, then transitions to CUT.
        """
        t0 = time.time()

        ok, seg_frame = self.cameras.read_seg()
        if not ok:
            log.warning("USB seg cam read failed.")
            time.sleep(0.1)
            return

        seg_result = self.seg.segment(seg_frame, state="SEGMENT")

        if not seg_result.success or seg_result.masks is None:
            is_fp = self.fp.record_stem_failure(state="SEGMENT")
            if is_fp:
                log.warning("Stem FP — marking POI box as FP zone, back to DETECT.")
                if self._poi_box:
                    self.fp.record_det_failure(self._poi_box, state="SEGMENT")
                self.detector.reset_confirmation()
                self._servo_start_time = 0.0
                self._transition(State.DETECT, reason="stem_fp")
            time.sleep(1.0 / self._seg_hz)
            return

        s_cfg = self.cfg["segmentation"]
        cp_res = extract_cutting_points(
            masks=seg_result.masks,
            method=s_cfg.get("cutting_point_method", "topmost"),
            min_stem_area_px=s_cfg.get("min_stem_area_px", 50),
            state="SEGMENT",
        )

        if cp_res.primary is None:
            self.fp.record_stem_failure(state="SEGMENT")
            time.sleep(1.0 / self._seg_hz)
            return

        primary = cp_res.primary
        cx, cy = primary.px, primary.py

        # ── Visual servoing: adjust end-effector servo ─────────────────────
        usb_h = self.cfg["cameras"]["usb_seg"].get("height", 720)
        target_y = usb_h // 2
        error_y = cy - target_y

        if abs(error_y) > 15:  # deadzone
            delta = error_y * 0.03
            if delta > 0: delta = min(3.0, max(1.0, delta))
            else:         delta = max(-3.0, min(-1.0, delta))
            self._servo_angle += delta
            self._servo_angle = max(0.0, min(44.0, self._servo_angle))

        angle = int(self._servo_angle)

        if time.time() - self._last_servo_send > 5.0 and abs(angle - self._last_sent_angle) >= 1:
            if angle > self._last_sent_angle:
                step_angle = self._last_sent_angle + 1
            else:
                step_angle = self._last_sent_angle - 1
            step_angle = max(0, min(44, step_angle))

            self.esp32._send({"cmd": "servo", "angle": step_angle})
            log.info(f"[SERVO] Step: {self._last_sent_angle}° → {step_angle}° "
                     f"(target={angle}°, err={error_y:.0f}px)")
            self._last_servo_send = time.time()
            self._last_sent_angle = step_angle

            # Start countdown on first servo send
            if self._servo_start_time == 0.0:
                self._servo_start_time = time.time()

        # Push seg data to Android
        self.ws.push_seg(seg_result.stem_count, cx, cy, None)

        # After servo_wait_sec seconds from first send → transition to CUT
        if self._servo_start_time > 0.0 and \
           time.time() - self._servo_start_time > self._servo_wait_sec:
            log.info("Servo settled. Transitioning to CUT (scissor).")
            self._state_data["cutting_point"] = primary
            self._cut_count = 0
            self._transition(State.CUT, reason="servo_settled")
            return

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0 / self._seg_hz - elapsed))

    def _tick_cut(self) -> None:
        """
        CUT — Trigger ESP32 scissor servo 3 times, then return to DETECT.
        """
        self._cut_count += 1
        log.info(f"✂ CUT {self._cut_count}/3")

        resp = self.esp32._send({"cmd": "cut"})
        if resp.ok:
            log.info(f"Cut {self._cut_count} done ({resp.latency_ms:.0f}ms)")
        else:
            log.warning(f"Cut {self._cut_count} failed: {resp.error}")

        if self._cut_count >= 3:
            log.info("Cut sequence complete. Resetting for next target.")

            # Reset servo to center
            self.esp32._send({"cmd": "servo", "angle": 22})
            self._servo_angle = 22.0
            self._last_sent_angle = -1
            self._servo_start_time = 0.0

            self.fp.reset_stem()
            self.detector.reset_confirmation()
            self._color_confirm_count = 0
            self._chassis_driving = False

            log_event("CUT", "cut_complete", "3x scissor cut done")
            time.sleep(1.0)
            self._transition(State.DETECT, reason="cut_done")
        else:
            time.sleep(1.0)  # pause between cuts

    def _tick_manual(self) -> None:
        self.ws.push_state("MANUAL", self._mode)
        time.sleep(0.1)

    def _tick_error(self) -> None:
        reason = self._state_data.get("reason", "unknown")
        log.error(f"ERROR state: {reason}")
        self.ws.push_state("ERROR", self._mode)
        self.ws.push_log("ERROR", reason)
        time.sleep(2.0)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _transition(self, new_state: State, reason: str = "") -> None:
        with self._state_lock:
            old = self._state.name
            self._state = new_state
            self._state_data["reason"] = reason
        log.info(f"State: {old} → {new_state.name}  ({reason})")
        log_event(new_state.name, "state_transition", f"{old}→{new_state.name}", extra=reason)
        self.ws.push_state(new_state.name, self._mode)

    def _wait_stm32_arrive(self, timeout_sec: float = 10.0) -> bool:
        """Poll STM32 status until it reports 'arrived' or timeout."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            resp = self.stm32.query_status()
            if resp.ok and "arrived" in resp.msg.lower():
                return True
            time.sleep(0.2)
        return False

    def _color_detect(self, frame):
        """
        Color-based fallback detection (blue-purple target).
        Returns (box, centre) tuple or None.
        """
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._color_clahe.apply(l)
            lab = cv2.merge([l, a, b])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([130, 80, 80]), np.array([150, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 3000:
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2
                    self._color_confirm_count += 1
                    return (x, y, x + w, y + h), (cx, cy)

            self._color_confirm_count = 0
            return None
        except Exception as e:
            log.debug(f"Color fallback error: {e}")
            self._color_confirm_count = 0
            return None