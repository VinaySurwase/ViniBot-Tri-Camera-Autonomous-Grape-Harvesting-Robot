"""
comms/ws_server.py — WebSocket Server (Android ↔ Pi)
=====================================================
Runs on the Pi. Android app connects to ws://<pi-ip>:8765

Message protocol (JSON):

  Android → Pi:
    {"type": "mode",    "value": "auto"|"manual"}
    {"type": "move",    "dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-255}
    {"type": "cmd",     "value": "start"|"stop"|"home"|"reset"}
    {"type": "ping"}

  Pi → Android:
    {"type": "state",   "state": "IDLE"|"DETECT"|"SEGMENT"|..., "mode": "auto"|"manual"}
    {"type": "coords",  "coord_type": "poi"|"cut",
                        "x": f, "y": f, "z": f,
                        "conf": f, "method": str, "stem_idx": int}
    {"type": "detection","found": bool, "cluster_box": [x1,y1,x2,y2]|null,
                          "conf": f, "method": str, "circles": int}
    {"type": "seg",     "stem_count": int, "cut_x": f, "cut_y": f,
                          "depth_mm": f|null}
    {"type": "log",     "level": "INFO"|"WARN"|"ERROR", "msg": str}
    {"type": "pong"}
    {"type": "error",   "msg": str}
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Optional, Set

log = logging.getLogger("ws_server")

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False
    log.warning("websockets not installed — Android comms disabled. pip install websockets")


class WSServer:
    """
    WebSocket server that bridges Android app commands to the state machine
    and pushes live state/coord data back to all connected clients.

    Thread-safe: state machine calls push_* from its own thread;
    asyncio event loop runs in a background thread.
    """

    def __init__(self, cfg: dict):
        ws = cfg["websocket"]
        self.host = ws.get("host", "0.0.0.0")
        self.port = int(ws.get("port", 8765))
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks registered by state machine
        self._on_mode_change:  Optional[Callable[[str], None]] = None
        self._on_move_cmd:     Optional[Callable[[str, int], None]] = None
        self._on_control_cmd:  Optional[Callable[[str], None]] = None

    # ── Registration ──────────────────────────────────────────────────────────

    def on_mode_change(self, cb: Callable[[str], None]):
        """cb(mode: str) called when Android sends mode change."""
        self._on_mode_change = cb

    def on_move_cmd(self, cb: Callable[[str, int], None]):
        """cb(direction: str, speed: int) called for manual move commands."""
        self._on_move_cmd = cb

    def on_control_cmd(self, cb: Callable[[str], None]):
        """cb(cmd: str) called for start/stop/home/reset."""
        self._on_control_cmd = cb

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not _HAS_WS:
            log.error("Cannot start WS server — websockets not installed.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-server")
        self._thread.start()
        log.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            log.info(f"WS serving on {self.host}:{self.port}")
            while self._running:
                await asyncio.sleep(0.1)

    # ── Handler ───────────────────────────────────────────────────────────────

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        client_addr = ws.remote_address
        log.info(f"Android client connected: {client_addr}")
        try:
            async for raw in ws:
                await self._handle_message(raw, ws)
        except Exception as e:
            log.debug(f"WS client {client_addr} disconnected: {e}")
        finally:
            self._clients.discard(ws)
            log.info(f"Android client disconnected: {client_addr}")

    async def _handle_message(self, raw: str, ws) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"type": "error", "msg": "invalid json"}))
            return

        mtype = msg.get("type", "")
        log.debug(f"Android → Pi: {msg}")

        if mtype == "ping":
            await ws.send(json.dumps({"type": "pong"}))

        elif mtype == "mode":
            mode = msg.get("value", "manual")
            log.info(f"Android set mode: {mode}")
            if self._on_mode_change:
                self._on_mode_change(mode)

        elif mtype == "move":
            direction = msg.get("dir", "stop")
            # Completely ignore whatever speed the app sends and force safe defaults
            speed = 30 if direction in ("left", "right") else 50
            
            log.debug(f"Android move: {direction} speed={speed} (forced)")
            if self._on_move_cmd:
                self._on_move_cmd(direction, speed)

        elif mtype == "cmd":
            cmd = msg.get("value", "")
            log.info(f"Android cmd: {cmd}")
            if self._on_control_cmd:
                self._on_control_cmd(cmd)

        else:
            await ws.send(json.dumps({"type": "error", "msg": f"unknown type: {mtype}"}))

    # ── Push methods (called from state machine thread) ───────────────────────

    def _push_sync(self, payload: dict) -> None:
        """Thread-safe push to all connected clients."""
        if not self._loop or not self._clients:
            return
        raw = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(raw), self._loop)

    async def _broadcast(self, raw: str) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(raw)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def push_state(self, state: str, mode: str) -> None:
        self._push_sync({"type": "state", "state": state, "mode": mode})

    def push_coords_poi(self, x: float, y: float, z: float,
                        conf: float = 0.0, method: str = "") -> None:
        self._push_sync({
            "type": "coords", "coord_type": "poi",
            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
            "conf": round(conf, 3), "method": method,
        })

    def push_coords_cut(self, x: float, y: float, z: Optional[float],
                        stem_idx: int = 0) -> None:
        self._push_sync({
            "type": "coords", "coord_type": "cut",
            "x": round(x, 2), "y": round(y, 2),
            "z": round(z, 2) if z else None,
            "stem_idx": stem_idx,
        })

    def push_detection(self, found: bool, cluster_box, conf: float,
                       method: str, circles: int) -> None:
        self._push_sync({
            "type": "detection",
            "found": found,
            "cluster_box": list(cluster_box) if cluster_box else None,
            "conf": round(conf, 3),
            "method": method,
            "circles": circles,
        })

    def push_seg(self, stem_count: int, cut_x: float, cut_y: float,
                 depth_mm: Optional[float]) -> None:
        self._push_sync({
            "type": "seg",
            "stem_count": stem_count,
            "cut_x": round(cut_x, 2),
            "cut_y": round(cut_y, 2),
            "depth_mm": round(depth_mm, 1) if depth_mm else None,
        })

    def push_log(self, level: str, msg: str) -> None:
        self._push_sync({"type": "log", "level": level, "msg": msg})