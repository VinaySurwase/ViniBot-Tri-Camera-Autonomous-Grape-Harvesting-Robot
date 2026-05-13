#!/usr/bin/env python3
"""
tools/test_scissor.py — Scissor Servo Calibration & Test
=========================================================
Interactive tool to determine open/close direction and test the cut action.

Controls:
  o     — Move to 0°  (test if this is OPEN or CLOSED)
  c     — Move to 70° (test if this is OPEN or CLOSED)
  u/d   — Nudge angle UP (+5°) or DOWN (-5°)
  CUT   — Press 'x' to trigger the full cut cycle
  q     — Quit

Usage:
  python tools/test_scissor.py
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from comms.esp32_comms import ESP32Comms

try:
    import getch
except ImportError:
    class Getch:
        def __call__(self):
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
    getch = Getch()

def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    esp = ESP32Comms(cfg)

    print("\n" + "=" * 60)
    print("  SCISSOR SERVO CALIBRATION & TEST")
    print("=" * 60)

    if not esp.connect():
        print("  ✗ Failed to connect to ESP32.")
        sys.exit(1)

    esp.set_mode("manual")

    current_angle = 0

    def send_scissor(angle):
        nonlocal current_angle
        angle = max(0, min(70, angle))
        current_angle = angle
        resp = esp._send({"cmd": "scissor", "angle": angle})
        status = "OK" if resp.ok else f"FAIL: {resp.error}"
        print(f"\r  Scissor → {angle:>3}°  [{status}]" + " " * 20, end="", flush=True)

    def send_cut():
        print(f"\n  ✂  Triggering CUT cycle (0→70→hold→0)...")
        resp = esp._send({"cmd": "cut"})
        if resp.ok:
            print(f"  ✓  Cut complete. ({resp.latency_ms:.0f}ms)")
        else:
            print(f"  ✗  Cut failed: {resp.error}")

    print()
    print("  STEP 1: Determine which direction is OPEN vs CLOSED")
    print("  ─────────────────────────────────────────────────────")
    print("  Press [o] to go to 0°.   Watch: are the blades OPEN or CLOSED?")
    print("  Press [c] to go to 70°.  Watch: are the blades OPEN or CLOSED?")
    print()
    print("  If 0°=OPEN and 70°=CLOSED → Current firmware is correct! ✓")
    print("  If 0°=CLOSED and 70°=OPEN → We need to swap the defines.")
    print()
    print("  Other controls:")
    print("    [u] / [d]  : Nudge +5° / -5°")
    print("    [1]-[7]    : Jump to 10°, 20°, 30°, 40°, 50°, 60°, 70°")
    print("    [x]        : Full CUT cycle")
    print("    [q]        : Quit")
    print("=" * 60)
    print()

    # Start at 0°
    send_scissor(0)
    print()

    try:
        while True:
            char = getch().lower()

            if char == 'q':
                print("\n\n  Quitting...")
                send_scissor(0)  # safe open
                break

            elif char == 'o':
                send_scissor(0)

            elif char == 'c':
                send_scissor(70)

            elif char == 'u':
                send_scissor(current_angle + 5)

            elif char == 'd':
                send_scissor(current_angle - 5)

            elif char == 'x':
                send_cut()

            elif char.isdigit() and char != '0':
                val = int(char) * 10
                send_scissor(val)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        send_scissor(0)
    finally:
        esp.close()
        print("\n  ✓ Closed.\n")

if __name__ == "__main__":
    main()
