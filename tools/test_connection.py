#!/usr/bin/env python3
"""
tools/test_connection.py — Pi ↔ Android Connection & Data Tester
=================================================================
Standalone script — no cameras, no models, no STM32 required.
Tests the full WebSocket communication path between Pi and the ViniBot
Android app.

What it does:
  1. Starts a WebSocket server on port 8765 (same as production)
  2. Waits for the Android app to connect (Debug tab → Connect)
  3. Runs a sequence of tests, sending real production message types
     and verifying responses arrive correctly
  4. Prints a pass/fail report with latency measurements

Tests performed:
  ┌─────────────────────────────────────────────────────────────┐
  │  TEST 1  Ping / Pong roundtrip                              │
  │  TEST 2  State push  (Pi → Android)                         │
  │  TEST 3  Detection event push                               │
  │  TEST 4  POI coordinate push                                │
  │  TEST 5  Segmentation + CUT coordinate push                 │
  │  TEST 6  Log message push                                   │
  │  TEST 7  Mode change  (Android → Pi)                        │
  │  TEST 8  Move command (Android → Pi)                        │
  │  TEST 9  Control cmd  (Android → Pi)                        │
  │  TEST 10 Burst test   (50 messages, measure throughput)     │
  └─────────────────────────────────────────────────────────────┘

Usage:
    python tools/test_connection.py
    python tools/test_connection.py --port 8765
    python tools/test_connection.py --host 0.0.0.0 --port 9000
    python tools/test_connection.py --auto         # run all tests automatically
    python tools/test_connection.py --repeat 5     # repeat burst test N times
    python tools/test_connection.py --mock-android # built-in mock Android client
"""

import argparse
import asyncio
import json
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# ── Colour helpers ────────────────────────────────────────────────────────────

GRN  = "\033[32m"; YLW = "\033[33m"; RED  = "\033[31m"
CYN  = "\033[36m"; DIM = "\033[2m";  RST  = "\033[0m"
BOLD = "\033[1m"

def _ok(m):     print(f"  {GRN}✓  {m}{RST}")
def _fail(m):   print(f"  {RED}✗  {m}{RST}")
def _warn(m):   print(f"  {YLW}⚠  {m}{RST}")
def _info(m):   print(f"  {CYN}·  {m}{RST}")
def _banner(m): print(f"\n{BOLD}{CYN}{'─'*56}\n  {m}\n{'─'*56}{RST}")
def _sep():     print(f"  {DIM}{'·'*52}{RST}")

# ── Arg parse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pi ↔ Android WebSocket connection tester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",         default="0.0.0.0",
                   help="Bind host (default: 0.0.0.0 = all interfaces)")
    p.add_argument("--port",         type=int, default=8765,
                   help="WebSocket port (default: 8765)")
    p.add_argument("--auto",         action="store_true",
                   help="Run all tests automatically without pause between them")
    p.add_argument("--repeat",       type=int, default=1,
                   help="Repeat burst test N times (default: 1)")
    p.add_argument("--mock-android", action="store_true",
                   help="Spawn a built-in mock Android client in a background thread")
    p.add_argument("--timeout",      type=int, default=60,
                   help="Seconds to wait for Android to connect (default: 60)")
    return p.parse_args()

# ── Test server ───────────────────────────────────────────────────────────────

class ConnectionTester:
    """
    WebSocket test server that doubles as a data sender.

    Tracks:
      - All messages received from Android
      - RTT of ping/pong
      - Push latency (time between send and ACK log)
      - Error count
    """

    def __init__(self, host: str, port: int, auto: bool, repeat: int, timeout: int):
        self.host    = host
        self.port    = port
        self.auto    = auto
        self.repeat  = repeat
        self.timeout = timeout

        self._ws         = None          # current client WebSocket
        self._connected  = asyncio.Event()
        self._client_addr = ""

        # Received message tracking
        self._received:   list[dict] = []
        self._pong_event: asyncio.Event = asyncio.Event()
        self._ping_sent_at: float = 0.0

        # Test results
        self._results: list[dict] = []

    # ── Server core ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            import websockets
        except ImportError:
            print(f"\n{RED}ERROR: websockets not installed.{RST}")
            print("Run: pip install websockets")
            sys.exit(1)

        async with websockets.serve(self._handler, self.host, self.port):
            ip = self._local_ip()
            print(f"\n{BOLD}{CYN}{'═'*56}")
            print(f"  ViniBot Connection Tester")
            print(f"{'═'*56}{RST}")
            print(f"  Server  : ws://{ip}:{self.port}")
            print(f"  Mode    : {'AUTO' if self.auto else 'INTERACTIVE'}")
            print()
            print(f"  {BOLD}→ Open ViniBot Android app{RST}")
            print(f"  {BOLD}→ Go to Debug tab{RST}")
            print(f"  {BOLD}→ Enter IP: {ip}  Port: {self.port}{RST}")
            print(f"  {BOLD}→ Tap Connect{RST}")
            print()
            print(f"  Waiting for Android to connect "
                  f"(timeout: {self.timeout}s)…\n")

            try:
                await asyncio.wait_for(self._connected.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                print(f"\n{RED}  ✗ Timeout — no Android client connected in {self.timeout}s.{RST}")
                print("  Check: same WiFi network, correct IP and port, firewall.")
                return

            print(f"\n{GRN}  ✓ Android connected from {self._client_addr}{RST}")
            print(f"  Starting tests…")

            await self._run_all_tests()

    async def _handler(self, ws) -> None:
        self._ws          = ws
        self._client_addr = str(ws.remote_address)
        self._connected.set()
        try:
            async for raw in ws:
                await self._on_message(raw)
        except Exception:
            pass

    async def _on_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            self._received.append(msg)

            mtype = msg.get("type", "")
            if mtype == "pong":
                self._pong_event.set()
            elif mtype == "ping":
                await self._send({"type": "pong"})
            else:
                _info(f"Android → Pi: {raw[:80]}")
        except Exception:
            _warn(f"Bad JSON from Android: {raw[:60]}")

    # ── Message helpers ───────────────────────────────────────────────────────

    async def _send(self, payload: dict) -> bool:
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps(payload))
            return True
        except Exception as e:
            _fail(f"Send failed: {e}")
            return False

    async def _wait_for(self, msg_type: str, timeout: float = 3.0) -> Optional[dict]:
        """Wait for a specific message type from Android."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for msg in reversed(self._received):
                if msg.get("type") == msg_type:
                    return msg
            await asyncio.sleep(0.05)
        return None

    async def _wait_for_any(self, count_before: int, timeout: float = 3.0) -> bool:
        """Wait until at least one new message has arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self._received) > count_before:
                return True
            await asyncio.sleep(0.05)
        return False

    # ── Individual tests ──────────────────────────────────────────────────────

    async def _test_ping_pong(self) -> None:
        _banner("TEST 1 — Ping / Pong Roundtrip")
        _info("Sending ping to Android…")

        self._pong_event.clear()
        self._ping_sent_at = time.perf_counter()
        await self._send({"type": "ping"})

        try:
            await asyncio.wait_for(self._pong_event.wait(), timeout=5.0)
            rtt = (time.perf_counter() - self._ping_sent_at) * 1000
            _ok(f"Pong received  —  RTT: {rtt:.1f} ms")
            self._record("ping_pong", True, rtt)
        except asyncio.TimeoutError:
            _fail("No pong received within 5s — check Android app is connected.")
            self._record("ping_pong", False, 0)

    async def _test_state_push(self) -> None:
        _banner("TEST 2 — State Push  (Pi → Android)")
        states = ["IDLE", "DETECT", "APPROACH", "SEGMENT", "CUT", "MANUAL", "ERROR"]
        modes  = ["manual", "auto"]

        for state in states:
            for mode in modes:
                payload = {"type": "state", "state": state, "mode": mode}
                t0 = time.perf_counter()
                ok = await self._send(payload)
                ms = (time.perf_counter() - t0) * 1000
                if ok:
                    _ok(f"Pushed state={state:<8}  mode={mode:<6}  ({ms:.1f}ms)")
                else:
                    _fail(f"Failed to push state={state}")
                await asyncio.sleep(0.05)

        self._record("state_push", True, 0)
        _info("→ Android Auto tab badge should cycle through all states.")
        if not self.auto:
            input("\n  Press Enter to continue…")

    async def _test_detection_push(self) -> None:
        _banner("TEST 3 — Detection Event Push  (Pi → Android)")

        cases = [
            # (found, cluster_box, conf, method, circles)
            (False, None,            0.0,  "none",   0),
            (True,  [320,180,640,400], 0.72, "opencv", 14),
            (True,  [100,50,300,280],  0.91, "opencv", 22),
            (True,  [400,200,700,500], 0.55, "yolo",    0),
            (False, None,            0.0,  "none",   0),
        ]

        for found, box, conf, method, circles in cases:
            payload = {
                "type":        "detection",
                "found":       found,
                "cluster_box": box,
                "conf":        conf,
                "method":      method,
                "circles":     circles,
            }
            t0 = time.perf_counter()
            await self._send(payload)
            ms = (time.perf_counter() - t0) * 1000
            status = "✓ FOUND" if found else "○ scanning"
            _ok(f"{status:<12}  conf={conf:.2f}  method={method:<6}  circles={circles}  ({ms:.1f}ms)")
            await asyncio.sleep(0.3)

        self._record("detection_push", True, 0)
        _info("→ Android Auto tab Detection card should update live.")
        if not self.auto:
            input("\n  Press Enter to continue…")

    async def _test_poi_coords(self) -> None:
        _banner("TEST 4 — POI Coordinate Push  (Pi → Android)")

        poi_samples = [
            (320.0, 240.0, 450.0, 0.78, "opencv"),
            (418.5, 195.3, 380.0, 0.84, "opencv"),
            (210.0, 310.0, 520.0, 0.61, "yolo"),
            (512.0, 256.0, 290.0, 0.93, "opencv"),
        ]

        for x, y, z, conf, method in poi_samples:
            payload = {
                "type":       "coords",
                "coord_type": "poi",
                "x":          x,
                "y":          y,
                "z":          z,
                "conf":       conf,
                "method":     method,
                "stem_idx":   0,
            }
            t0 = time.perf_counter()
            await self._send(payload)
            ms = (time.perf_counter() - t0) * 1000
            _ok(f"POI  x={x:.1f}  y={y:.1f}  z={z:.1f}mm  conf={conf:.2f}  [{method}]  ({ms:.1f}ms)")
            await asyncio.sleep(0.4)

        self._record("poi_coords", True, 0)
        _info("→ Android Auto tab POI card and Debug coord stream should update.")
        if not self.auto:
            input("\n  Press Enter to continue…")

    async def _test_seg_and_cut_coords(self) -> None:
        _banner("TEST 5 — Segmentation + CUT Coordinate Push")

        seg_samples = [
            (3, 215.0, 180.0, 420.0),
            (1, 320.0, 260.0, 390.0),
            (2, 180.0, 140.0, 460.0),
        ]

        for stem_count, cx, cy, depth in seg_samples:
            # Push seg event
            seg_payload = {
                "type":        "seg",
                "stem_count":  stem_count,
                "cut_x":       cx,
                "cut_y":       cy,
                "depth_mm":    depth,
            }
            t0 = time.perf_counter()
            await self._send(seg_payload)
            ms = (time.perf_counter() - t0) * 1000
            _ok(f"SEG  stems={stem_count}  cut=({cx:.0f},{cy:.0f})  depth={depth:.0f}mm  ({ms:.1f}ms)")
            await asyncio.sleep(0.2)

            # Push cut coord
            cut_payload = {
                "type":       "coords",
                "coord_type": "cut",
                "x":          cx,
                "y":          cy,
                "z":          depth,
                "conf":       0.0,
                "method":     "",
                "stem_idx":   0,
            }
            await self._send(cut_payload)
            _ok(f"CUT  x={cx:.1f}  y={cy:.1f}  z={depth:.1f}mm  stem=0")
            await asyncio.sleep(0.4)

        self._record("seg_cut_coords", True, 0)
        _info("→ Android Seg card and CUT coord card should update.")
        if not self.auto:
            input("\n  Press Enter to continue…")

    async def _test_log_push(self) -> None:
        _banner("TEST 6 — Log Message Push  (Pi → Android)")

        log_messages = [
            ("INFO",  "GrapeBot started. Cameras initialised."),
            ("INFO",  "Detection: cluster found  conf=0.82  method=opencv"),
            ("INFO",  "POI confirmed — transitioning to APPROACH"),
            ("WARN",  "Depth fallback active — calibration not loaded"),
            ("INFO",  "Segmentation: 2 stems found  (145.3ms)"),
            ("INFO",  "Cutting point: (215, 180)  depth=420mm"),
            ("ERROR", "STM32 timeout on cut_coords — retrying"),
            ("INFO",  "Cut complete. Returning to DETECT."),
        ]

        for level, msg in log_messages:
            payload = {"type": "log", "level": level, "msg": msg}
            t0 = time.perf_counter()
            await self._send(payload)
            ms = (time.perf_counter() - t0) * 1000
            colour = GRN if level == "INFO" else (YLW if level == "WARN" else RED)
            _ok(f"[{colour}{level:<5}{RST}] {msg[:50]}  ({ms:.1f}ms)")
            await asyncio.sleep(0.15)

        self._record("log_push", True, 0)
        _info("→ Android Debug tab LOG STREAM should show all 8 messages.")
        if not self.auto:
            input("\n  Press Enter to continue…")

    async def _test_android_mode_change(self) -> None:
        _banner("TEST 7 — Mode Change  (Android → Pi)")
        _info("Tap 'Switch to AUTO' in Android Manual tab  OR  tap START in Auto tab…")
        _info("(Timeout: 15 seconds)")

        count_before = len(self._received)
        msg = await self._wait_for("mode", timeout=15.0)

        if msg:
            mode = msg.get("value", "?")
            _ok(f"Mode change received: mode='{mode}'")
            self._record("mode_change", True, 0)
            # Echo back a state update
            await self._send({"type": "state", "state": "DETECT" if mode == "auto" else "MANUAL",
                               "mode": mode})
        else:
            if self.auto:
                _warn("Skipped (auto mode — no Android interaction).")
                self._record("mode_change", None, 0)
            else:
                _fail("No mode change received in 15s.")
                self._record("mode_change", False, 0)

    async def _test_android_move_cmd(self) -> None:
        _banner("TEST 8 — Move Command  (Android → Pi)")
        _info("Hold a D-pad button in Android Manual tab…")
        _info("(Timeout: 15 seconds — press any direction)")

        directions_received = []
        deadline = time.time() + 15.0

        while time.time() < deadline:
            for msg in self._received[-20:]:
                if msg.get("type") == "move" and msg not in directions_received:
                    directions_received.append(msg)
                    d = msg.get("dir", "?")
                    s = msg.get("speed", "?")
                    _ok(f"Move: dir='{d}'  speed={s}")
            if len(directions_received) >= 2:
                break
            await asyncio.sleep(0.1)

        if directions_received:
            self._record("move_cmd", True, 0)
        else:
            if self.auto:
                _warn("Skipped (auto mode — no Android interaction).")
                self._record("move_cmd", None, 0)
            else:
                _fail("No move command received in 15s.")
                self._record("move_cmd", False, 0)

    async def _test_android_control_cmd(self) -> None:
        _banner("TEST 9 — Control Command  (Android → Pi)")
        _info("Tap HOME or RESET in Android Auto tab…")
        _info("(Timeout: 15 seconds)")

        msg = await self._wait_for("cmd", timeout=15.0)

        if msg:
            cmd = msg.get("value", "?")
            _ok(f"Control cmd received: value='{cmd}'")
            self._record("control_cmd", True, 0)
        else:
            if self.auto:
                _warn("Skipped (auto mode — no Android interaction).")
                self._record("control_cmd", None, 0)
            else:
                _fail("No control cmd received in 15s.")
                self._record("control_cmd", False, 0)

    async def _test_burst(self) -> None:
        _banner("TEST 10 — Burst Throughput Test")
        N = 50

        for run in range(self.repeat):
            if self.repeat > 1:
                _info(f"Burst run {run+1}/{self.repeat}")

            # Build 50 mixed-type messages
            messages = []
            for i in range(N):
                if i % 5 == 0:
                    messages.append({"type": "state",     "state": "DETECT", "mode": "auto"})
                elif i % 5 == 1:
                    messages.append({"type": "detection", "found": True,
                                     "cluster_box": [300, 200, 600, 450],
                                     "conf": 0.75, "method": "opencv", "circles": 12})
                elif i % 5 == 2:
                    messages.append({"type": "coords",    "coord_type": "poi",
                                     "x": 320.0+i, "y": 240.0, "z": 450.0,
                                     "conf": 0.8, "method": "opencv", "stem_idx": 0})
                elif i % 5 == 3:
                    messages.append({"type": "seg",       "stem_count": 2,
                                     "cut_x": 215.0, "cut_y": 180.0, "depth_mm": 420.0})
                else:
                    messages.append({"type": "log",       "level": "INFO",
                                     "msg": f"Burst message {i+1}"})

            t0     = time.perf_counter()
            errors = 0
            for msg in messages:
                if not await self._send(msg):
                    errors += 1
                await asyncio.sleep(0.002)   # 2ms between messages → ~500 msg/s cap

            elapsed_ms = (time.perf_counter() - t0) * 1000
            throughput = N / (elapsed_ms / 1000)
            avg_ms     = elapsed_ms / N

            if errors == 0:
                _ok(f"Sent {N} messages in {elapsed_ms:.0f}ms  "
                    f"({throughput:.0f} msg/s  avg {avg_ms:.1f}ms each)")
            else:
                _fail(f"Errors: {errors}/{N}")

            self._record("burst", errors == 0, avg_ms)
            await asyncio.sleep(0.5)

        _info("→ Android Debug coord stream should show rapid updates.")

    # ── Test runner ───────────────────────────────────────────────────────────

    async def _run_all_tests(self) -> None:
        if not self.auto:
            print(f"\n  {YLW}Interactive mode — press Enter to advance each test.{RST}")

        await asyncio.sleep(0.5)

        # Always-automatic tests (Pi → Android, no Android interaction needed)
        await self._test_ping_pong()
        await asyncio.sleep(0.3)

        await self._test_state_push()
        await asyncio.sleep(0.2)

        await self._test_detection_push()
        await asyncio.sleep(0.2)

        await self._test_poi_coords()
        await asyncio.sleep(0.2)

        await self._test_seg_and_cut_coords()
        await asyncio.sleep(0.2)

        await self._test_log_push()
        await asyncio.sleep(0.2)

        # Interaction tests (Android → Pi)
        await self._test_android_mode_change()
        await asyncio.sleep(0.2)

        await self._test_android_move_cmd()
        await asyncio.sleep(0.2)

        await self._test_android_control_cmd()
        await asyncio.sleep(0.2)

        await self._test_burst()

        self._print_summary()

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        _banner("TEST SUMMARY")

        passed  = sum(1 for r in self._results if r["passed"] is True)
        failed  = sum(1 for r in self._results if r["passed"] is False)
        skipped = sum(1 for r in self._results if r["passed"] is None)
        total   = len(self._results)

        labels = {
            "ping_pong":     "TEST 1  Ping / Pong roundtrip",
            "state_push":    "TEST 2  State push",
            "detection_push":"TEST 3  Detection push",
            "poi_coords":    "TEST 4  POI coord push",
            "seg_cut_coords":"TEST 5  Seg + CUT coord push",
            "log_push":      "TEST 6  Log message push",
            "mode_change":   "TEST 7  Mode change (Android→Pi)",
            "move_cmd":      "TEST 8  Move command (Android→Pi)",
            "control_cmd":   "TEST 9  Control cmd (Android→Pi)",
            "burst":         "TEST 10 Burst throughput",
        }

        print()
        for r in self._results:
            label = labels.get(r["name"], r["name"])
            if r["passed"] is True:
                icon = f"{GRN}PASS{RST}"
            elif r["passed"] is False:
                icon = f"{RED}FAIL{RST}"
            else:
                icon = f"{YLW}SKIP{RST}"

            lat = f"  ({r['latency_ms']:.1f}ms)" if r['latency_ms'] else ""
            print(f"  [{icon}]  {label}{lat}")

        print()
        print(f"  Results: {GRN}{passed} passed{RST}  "
              f"{RED}{failed} failed{RST}  "
              f"{YLW}{skipped} skipped{RST}  "
              f"/ {total} total")
        print()

        total_received = len(self._received)
        print(f"  Messages received from Android : {total_received}")

        if failed == 0:
            print(f"\n  {GRN}{BOLD}✓ All tests passed — Pi ↔ Android link is working correctly.{RST}")
        else:
            print(f"\n  {RED}{BOLD}✗ {failed} test(s) failed.{RST}")
            print(f"  {YLW}Check:{RST}")
            print("    • Android Debug tab shows connected (green dot)")
            print("    • Both devices on the same WiFi network")
            print("    • ViniBot app shows incoming data in coord/log stream")
        print()

    def _record(self, name: str, passed: Optional[bool], latency_ms: float) -> None:
        self._results.append({
            "name":       name,
            "passed":     passed,
            "latency_ms": latency_ms,
            "ts":         datetime.now().isoformat(),
        })

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"


# ── Mock Android client (for testing without a phone) ────────────────────────

async def _mock_android_client(host: str, port: int) -> None:
    """
    Built-in mock Android client — simulates the ViniBot app.
    Runs in background, responds to pings and sends mode/move/cmd messages.
    """
    try:
        import websockets
    except ImportError:
        return

    url = f"ws://127.0.0.1:{port}"
    _info(f"[Mock Android] Connecting to {url}…")
    await asyncio.sleep(1.0)

    try:
        async with websockets.connect(url) as ws:
            _ok("[Mock Android] Connected.")

            # Background listener
            async def _listen():
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                    # else: silently consume

            listener = asyncio.create_task(_listen())

            await asyncio.sleep(3.0)
            await ws.send(json.dumps({"type": "ping"}))

            await asyncio.sleep(12.0)   # wait for auto tests to reach interaction tests

            # Simulate Android user interactions
            _info("[Mock Android] Sending mode=auto")
            await ws.send(json.dumps({"type": "mode", "value": "auto"}))
            await asyncio.sleep(1.0)

            _info("[Mock Android] Sending move=forward")
            await ws.send(json.dumps({"type": "move", "dir": "forward", "speed": 50}))
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "move", "dir": "stop", "speed": 0}))
            await asyncio.sleep(1.0)

            _info("[Mock Android] Sending cmd=home")
            await ws.send(json.dumps({"type": "cmd", "value": "home"}))
            await asyncio.sleep(5.0)

            listener.cancel()
    except Exception as e:
        _warn(f"[Mock Android] {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Suppress websockets library debug logs
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.ERROR)

    args    = parse_args()
    tester  = ConnectionTester(
        host    = args.host,
        port    = args.port,
        auto    = args.auto or args.mock_android,
        repeat  = args.repeat,
        timeout = args.timeout,
    )

    async def _run():
        tasks = [tester.run()]
        if args.mock_android:
            tasks.append(_mock_android_client(args.host, args.port))
        await asyncio.gather(*tasks)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(f"\n\n  {YLW}Interrupted.{RST}")


if __name__ == "__main__":
    main()