"""
comms/stm32_comms.py — STM32 Bot & Arm Motion Controller
=========================================================
Sends movement and coordinate commands to the STM32 over UART serial.

STM32 is the primary motion controller for:
  • Bot chassis (4-wheel drive) — forward / backward / left / right / stop
  • Robotic arm — move to POI, move to cutting point, actuate cutter, home

Protocol — dual-mode over UART:
  Coordinates (char-by-char plain text — proven reliable, matches temp/uart_send.py):
    "x.x#y.y#z.z\n"   — hash-separated, newline-terminated
    Each char sent individually with configurable delay to prevent STM32 buffer overruns.

  Other commands (newline-terminated JSON):
    {"cmd": "move",   "dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-255}
    {"cmd": "status"}
    {"cmd": "stop"}
    {"cmd": "home"}
    {"cmd": "arm_enable",  "state": true|false}
    {"cmd": "cutter_fire"}

  STM32 → Pi:
    {"ok": true,  "msg": "..."}          — ACK
    {"ok": false, "error": "..."}        — NACK
    {"status": "idle"|"moving"|"arrived"|"error", "pos": [x,y,z]}  — status reply

Debug flag: when coord_debug is enabled in config every outgoing coordinate is
also written to coords.jsonl and the ring buffer for the Android debug panel.

UART wiring (Pi 5 → STM32):
  Pi GPIO 14 (TX)  →  STM32 RX  (e.g. USART1 PA10 or USART2 PA3)
  Pi GPIO 15 (RX)  →  STM32 TX
  Pi GND           →  STM32 GND
  (3.3 V logic — STM32 is 3.3 V tolerant on most pins)

  Alternatively use a USB-to-UART adapter → /dev/ttyUSB0 or /dev/ttyACM0.

Enable Pi hardware UART:
  sudo raspi-config → Interface Options → Serial Port
  → "Login shell accessible over serial?" NO
  → "Serial port hardware enabled?"      YES
  Reboot. Port will be /dev/ttyAMA0 (primary UART) or /dev/serial0.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

from logging_.logger import get_logger, log_event
from logging_.coord_debug import CoordDebugLogger

log = get_logger("stm32_comms")


# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class STM32Response:
    ok:         bool
    msg:        str   = ""
    error:      str   = ""
    latency_ms: float = 0.0
    status:     str   = ""          # "idle" | "moving" | "arrived" | "error"
    position:   list  = None        # [x, y, z] from status reply


# ── Main class ────────────────────────────────────────────────────────────────

class STM32Comms:
    """
    UART interface to the STM32 motion controller.

    Thread-safe: all sends are protected by a lock.
    Reconnects automatically on serial errors.

    In dry_run mode all commands are logged but never sent — safe for
    bench testing without hardware.
    """

    def __init__(self, cfg: dict, coord_debug: CoordDebugLogger):
        s = cfg["stm32"]
        self.dry_run        = s.get("dry_run", False)
        self.port           = s.get("serial_port",  "/dev/ttyAMA0")
        self.baud           = s.get("serial_baud",  115200)
        self.timeout_ms     = s.get("command_timeout_ms", 500)
        self.char_delay_s   = s.get("char_delay_ms", 50) / 1000.0
        self.warmup_s       = s.get("warmup_s", 2.0)
        self.coord_debug    = coord_debug
        self._serial        = None
        self._lock          = threading.Lock()
        self._connected     = False

        if self.dry_run:
            log.info("STM32 comms: DRY-RUN mode — no commands will be sent.")
        else:
            log.info(f"STM32 comms: port={self.port}  baud={self.baud}  "
                     f"timeout={self.timeout_ms}ms")

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if self.dry_run:
            self._connected = True
            return True
        try:
            import serial
            self._serial = serial.Serial(
                port     = self.port,
                baudrate = self.baud,
                timeout  = 1,
            )
            time.sleep(self.warmup_s)     # match proven uart_send.py warmup
            self._serial.reset_input_buffer()
            self._connected = True
            log.info(f"STM32 UART connected: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            log.error(f"STM32 connect failed: {e}")
            log.error("Check: port in config.yaml, raspi-config serial settings, wiring.")
            return False

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False
        log.info("STM32 UART closed.")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Bot motion commands ───────────────────────────────────────────────────

    def move(self, direction: str, speed: int = None) -> STM32Response:
        """
        Drive the bot chassis.
        direction: "forward" | "backward" | "left" | "right" | "stop"
        speed    : 0–255 PWM value sent to STM32 motor driver (default: 50 for drive, 30 for turn)
        """
        assert direction in ("forward", "backward", "left", "right", "stop"), \
            f"Invalid direction: {direction}"

        if speed is None:
            speed = 30 if direction in ("left", "right") else 50

        speed = int(max(0, min(255, speed)))
        log.debug(f"STM32 ← move {direction}  speed={speed}")
        return self._send({"cmd": "move", "dir": direction, "speed": speed})

    def stop(self) -> STM32Response:
        """Immediate stop — sends speed=0 stop command."""
        return self.move("stop", 0)

    def home(self) -> STM32Response:
        """Return arm to home position."""
        log.info("STM32 ← home")
        return self._send({"cmd": "home"})

    def arm_enable(self, enable: bool = True) -> STM32Response:
        """Enable or disable the robotic arm power rail."""
        log.info(f"STM32 ← arm_enable {enable}")
        return self._send({"cmd": "arm_enable", "state": enable})

    def cutter_fire(self) -> STM32Response:
        """Actuate the grape stem cutter (solenoid / servo)."""
        log.info("STM32 ← cutter_fire")
        return self._send({"cmd": "cutter_fire"})

    def query_status(self) -> STM32Response:
        """Query current motion status. Returns STM32Response with .status field."""
        return self._send({"cmd": "status"})

    # ── Coordinate commands ───────────────────────────────────────────────────

    def send_poi_coords(
        self,
        x: float, y: float, z: float,
        conf:   float = 0.0,
        method: str   = "",
        state:  str   = "",
    ) -> STM32Response:
        """
        Send grape cluster POI coordinates to STM32.
        x = horizontal angle (0=centre, +left, -right)
        y = depth in mm (from stereo disparity)
        z = height in arm coordinate system (0–325mm)
            Reference (0) is 390mm below camera centre.

        STM32 uses these to drive the bot chassis toward the cluster.
        """
        log.info(f"[COORD][POI] x={x:.1f} y={y:.1f} z={z:.1f}mm  "
                 f"conf={conf:.2f}  method={method}")
        self.coord_debug.log(
            coord_type="poi", x=x, y=y, z=z,
            conf=conf, method=method, state=state,
        )
        log_event(
            state=state, event="poi_coords_sent",
            value=f"x={x:.1f},y={y:.1f},z={z:.1f}",
            extra=f"conf={conf:.2f},method={method}",
        )
        return self._send_coords_raw(x, y, z)

    def send_cut_coords(
        self,
        x: float, y: float, z: float,
        stem_idx: int = 0,
        area_px:  int = 0,
        state:    str = "",
    ) -> STM32Response:
        """
        Send stem cutting point coordinates to STM32.
        x = horizontal angle (0=centre, +left, -right)
        y = depth in mm (from stereo disparity)
        z = height in arm coordinate system (0–325mm)
            Reference (0) is 390mm below camera centre.

        STM32 uses these to position the arm end-effector at the cut point.
        """
        log.info(f"[COORD][CUT] x={x:.1f} y={y:.1f} z={z:.1f}mm  "
                 f"stem={stem_idx}  area={area_px}px")
        self.coord_debug.log(
            coord_type="cut", x=x, y=y, z=z,
            stem_idx=stem_idx, area_px=area_px, state=state,
        )
        log_event(
            state=state, event="cut_coords_sent",
            value=f"x={x:.1f},y={y:.1f},z={z:.1f}",
            extra=f"stem={stem_idx},area={area_px}",
        )
        return self._send_coords_raw(x, y, z)

    # ── Char-by-char coordinate send (matches proven temp/uart_send.py) ────────

    def _send_coords_raw(self, x: float, y: float, z: float) -> STM32Response:
        """
        Send coordinates using the proven char-by-char plain-text protocol.
        Format: "x.x#y.y#z.z\n" — exactly matching temp/uart_send.py

        Each character is sent individually with a small delay between them
        to prevent STM32 UART buffer overruns.
        """
        if self.dry_run:
            coord_str = f"{x:.1f}#{y:.1f}#{z:.1f}"
            log.info(f"[DRY-RUN] STM32 ← coords: {coord_str}")
            time.sleep(0.005)
            return STM32Response(ok=True, msg="dry_run")

        # Auto-reconnect
        if not self._connected or self._serial is None or not self._serial.is_open:
            log.warning("STM32 not connected — attempting reconnect.")
            if not self.connect():
                return STM32Response(ok=False, error="not_connected")

        command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"
        t0 = time.perf_counter()

        with self._lock:
            try:
                log.debug(f"STM32 ← coords (char-by-char): {command_string.strip()}")

                # Send exactly one byte at a time — proven reliable
                for char in command_string:
                    self._serial.write(char.encode('utf-8'))
                    time.sleep(self.char_delay_s)

                latency = (time.perf_counter() - t0) * 1000.0
                log.info(f"STM32 coords transmitted in {latency:.0f}ms")
                return STM32Response(ok=True, msg="coords_sent", latency_ms=latency)

            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                log.error(f"STM32 UART error during coord send: {exc}")
                self._connected = False
                return STM32Response(ok=False, error=str(exc), latency_ms=latency)

    # ── Internal UART send / receive (JSON — for non-coordinate commands) ─────

    def _send(self, payload: dict) -> STM32Response:
        """
        Serialise payload to JSON, write over UART, read one-line response.
        Thread-safe via _lock.
        """
        # Dry-run: log and return synthetic OK
        if self.dry_run:
            log.info(f"[DRY-RUN] STM32 ← {json.dumps(payload)}")
            time.sleep(0.005)          # simulate tiny latency
            return STM32Response(ok=True, msg="dry_run")

        # Auto-reconnect
        if not self._connected or self._serial is None or not self._serial.is_open:
            log.warning("STM32 not connected — attempting reconnect.")
            if not self.connect():
                return STM32Response(ok=False, error="not_connected")

        raw = json.dumps(payload, separators=(",", ":")) + "\n"
        t0  = time.perf_counter()

        with self._lock:
            try:
                # ── Write ─────────────────────────────────────────────────
                self._serial.write(raw.encode("utf-8"))
                self._serial.flush()

                # ── Read response (one line, timeout via serial.timeout) ──
                reply_raw = self._serial.readline()
                latency   = (time.perf_counter() - t0) * 1000.0

                if not reply_raw:
                    log.warning(f"STM32 timeout — no response to: {raw.strip()}")
                    return STM32Response(ok=False, error="timeout", latency_ms=latency)

                reply_str = reply_raw.decode("utf-8", errors="replace").strip()
                log.debug(f"STM32 → {reply_str}  ({latency:.1f}ms)")

                # ── Parse reply ───────────────────────────────────────────
                return self._parse_reply(reply_str, latency)

            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                log.error(f"STM32 UART error: {exc}")
                self._connected = False
                return STM32Response(ok=False, error=str(exc), latency_ms=latency)

    def _parse_reply(self, raw: str, latency_ms: float) -> STM32Response:
        """
        Parse a JSON reply line from STM32.
        Handles both ACK replies and status replies gracefully.
        If the STM32 sends plain "OK" or "ERR:..." text, that is also handled.
        """
        # Plain-text fallback (some STM32 firmwares send simple strings)
        if raw.upper() in ("OK", "ACK"):
            return STM32Response(ok=True, msg=raw, latency_ms=latency_ms)
        if raw.upper().startswith("ERR") or raw.upper().startswith("NAK"):
            return STM32Response(ok=False, error=raw, latency_ms=latency_ms)

        # JSON reply
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"STM32 non-JSON reply: {raw!r}")
            # Treat any non-empty, non-error reply as OK
            return STM32Response(ok=True, msg=raw, latency_ms=latency_ms)

        # Status reply: {"status": "arrived", "pos": [x,y,z]}
        if "status" in data:
            return STM32Response(
                ok=True,
                status=data.get("status", ""),
                position=data.get("pos"),
                latency_ms=latency_ms,
            )

        # Standard ACK: {"ok": true, "msg": "..."}
        return STM32Response(
            ok=bool(data.get("ok", False)),
            msg=data.get("msg", ""),
            error=data.get("error", ""),
            latency_ms=latency_ms,
        )