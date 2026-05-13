"""
comms/esp32_comms.py — ESP32 Chassis Motor Controller
======================================================
Sends chassis drive commands to the ESP32 over UART serial.

ESP32 controls:
  • 4-wheel bot chassis (2× BTS7960B H-bridge, left/right side)
  • 3× HC-SR04 ultrasonic sensors (obstacle avoidance)

Protocol — newline-terminated JSON over UART:
  Pi → ESP32:
    {"cmd": "move",   "dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-255}
    {"cmd": "stop"}
    {"cmd": "mode",   "value": "auto"|"manual"}
    {"cmd": "status"}

  ESP32 → Pi:
    {"ok": true,  "msg": "moving_forward"}
    {"ok": false, "error": "obstacle_front"}
    {"ok": true,  "status": "idle", "mode": "manual", "speed": 0,
                  "sensors": {"front": 1200, "left": 800, "right": 750}}

UART wiring (RPi5 → ESP32):
  Pi GPIO 14 (TX, Pin 8)  →  ESP32 GPIO 16 (RX2)
  Pi GPIO 15 (RX, Pin 10) ←  ESP32 GPIO 17 (TX2)
  Pi GND (Pin 6)          ─── ESP32 GND
  Baud: 115200

Enable Pi hardware UART:
  sudo raspi-config → Interface Options → Serial Port
  → Login shell: NO
  → Serial hardware enabled: YES
  Reboot. Use /dev/ttyAMA0 or /dev/serial0.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

from logging_.logger import get_logger, log_event

log = get_logger("esp32_comms")


# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class ESP32Response:
    ok:          bool
    msg:         str            = ""
    error:       str            = ""
    latency_ms:  float          = 0.0
    status:      str            = ""   # "idle"|"moving_forward"|"turning_left" ...
    sensors:     Dict[str, int] = field(default_factory=dict)  # front/left/right mm


# ── Main class ────────────────────────────────────────────────────────────────

class ESP32Comms:
    """
    UART interface to the ESP32 chassis motor controller.

    Thread-safe: all sends are protected by a lock.
    Reconnects automatically on serial errors.

    In dry_run mode all commands are logged but never sent.
    """

    def __init__(self, cfg: dict):
        e = cfg.get("esp32", {})
        self.dry_run       = e.get("dry_run",           False)
        self.port          = e.get("serial_port",        "/dev/ttyAMA2")
        self.baud          = e.get("serial_baud",        115200)
        self.timeout_ms    = e.get("command_timeout_ms", 500)

        self._serial    = None
        self._lock      = threading.Lock()
        self._connected = False

        # Last known sensor readings — updated on every status reply
        self.last_sensors: Dict[str, int] = {"front": 9999, "left": 9999, "right": 9999}

        if self.dry_run:
            log.info("ESP32 comms: DRY-RUN mode — no commands will be sent.")
        else:
            log.info(f"ESP32 comms: port={self.port}  baud={self.baud}  "
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
                timeout  = self.timeout_ms / 1000.0,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
            )
            time.sleep(0.1)
            self._serial.reset_input_buffer()
            self._connected = True
            log.info(f"ESP32 UART connected: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            log.error(f"ESP32 connect failed: {e}")
            log.error("Check: port in config.yaml, raspi-config serial settings, wiring.")
            return False

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False
        log.info("ESP32 UART closed.")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Bot chassis commands ──────────────────────────────────────────────────

    def move(self, direction: str, speed: int = None) -> ESP32Response:
        """
        Drive the bot chassis.
        direction: "forward" | "backward" | "left" | "right" | "stop"
        speed    : 0–255 PWM value (default: 50 for forward/backward, 30 for turns)
        """
        assert direction in ("forward", "backward", "left", "right", "stop"), \
            f"Invalid direction: {direction}"
        
        if speed is None:
            speed = 30 if direction in ("left", "right") else 50

        speed = int(max(0, min(255, speed)))
        log.debug(f"ESP32 ← move {direction}  speed={speed}")
        return self._send({"cmd": "move", "dir": direction, "speed": speed})

    def stop(self) -> ESP32Response:
        """Immediate full stop."""
        log.info("ESP32 ← STOP")
        return self._send({"cmd": "stop"})

    def set_mode(self, mode: str) -> ESP32Response:
        """
        Set ESP32 drive mode.
        mode: "auto"   — obstacle avoidance active
              "manual" — direct command only
        """
        assert mode in ("auto", "manual"), f"Invalid mode: {mode}"
        log.info(f"ESP32 ← mode={mode}")
        return self._send({"cmd": "mode", "value": mode})

    def query_status(self) -> ESP32Response:
        """
        Request current status + sensor readings from ESP32.
        Sensor values are also stored in self.last_sensors.
        """
        return self._send({"cmd": "status"})

    # ── Internal UART send / receive ──────────────────────────────────────────

    def _send(self, payload: dict) -> ESP32Response:
        if self.dry_run:
            log.info(f"[DRY-RUN] ESP32 ← {json.dumps(payload)}")
            time.sleep(0.005)
            return ESP32Response(ok=True, msg="dry_run")

        # Auto-reconnect
        if not self._connected or self._serial is None or not self._serial.is_open:
            log.warning("ESP32 not connected — attempting reconnect.")
            if not self.connect():
                return ESP32Response(ok=False, error="not_connected")

        raw = json.dumps(payload, separators=(",", ":")) + "\n"
        t0  = time.perf_counter()

        with self._lock:
            try:
                self._serial.write(raw.encode("utf-8"))
                self._serial.flush()

                reply_raw = self._serial.readline()
                latency   = (time.perf_counter() - t0) * 1000.0

                if not reply_raw:
                    log.warning(f"ESP32 timeout — no response to: {raw.strip()}")
                    return ESP32Response(ok=False, error="timeout", latency_ms=latency)

                reply_str = reply_raw.decode("utf-8", errors="replace").strip()
                log.debug(f"ESP32 → {reply_str}  ({latency:.1f}ms)")
                return self._parse_reply(reply_str, latency)

            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                log.error(f"ESP32 UART error: {exc}")
                self._connected = False
                return ESP32Response(ok=False, error=str(exc), latency_ms=latency)

    def _parse_reply(self, raw: str, latency_ms: float) -> ESP32Response:
        if raw.upper() in ("OK", "ACK"):
            return ESP32Response(ok=True, msg=raw, latency_ms=latency_ms)
        if raw.upper().startswith("ERR"):
            return ESP32Response(ok=False, error=raw, latency_ms=latency_ms)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"ESP32 non-JSON reply: {raw!r}")
            return ESP32Response(ok=True, msg=raw, latency_ms=latency_ms)

        sensors = data.get("sensors", {})
        if sensors:
            self.last_sensors = {
                "front": int(sensors.get("front", 9999)),
                "left":  int(sensors.get("left",  9999)),
                "right": int(sensors.get("right", 9999)),
            }

        return ESP32Response(
            ok         = bool(data.get("ok", False)),
            msg        = data.get("msg",    ""),
            error      = data.get("error",  ""),
            status     = data.get("status", ""),
            sensors    = self.last_sensors.copy(),
            latency_ms = latency_ms,
        )
