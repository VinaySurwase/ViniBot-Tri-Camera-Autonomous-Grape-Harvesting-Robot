#!/usr/bin/env python3
"""
tools/test_esp32_comms.py
=========================
Tests UART communication with the ESP32 chassis controller.
Sends move commands and prints the ESP32's response.
Run this with the ESP32 connected via UART before deploying the full bot.

Usage:
  python tools/test_esp32_comms.py
  python tools/test_esp32_comms.py --dry-run   # simulate without hardware
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml

def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simulate without hardware")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.dry_run:
        cfg.setdefault("esp32", {})["dry_run"] = True

    from comms.esp32_comms import ESP32Comms
    esp = ESP32Comms(cfg)

    print("\n" + "=" * 55)
    print("  ESP32 Chassis Communication Test")
    print("=" * 55)
    print(f"  Port    : {esp.port}")
    print(f"  Baud    : {esp.baud}")
    print(f"  Dry Run : {esp.dry_run}")
    print("=" * 55 + "\n")

    if not esp.connect():
        print("  ✗ Failed to connect to ESP32. Check UART wiring and port.")
        sys.exit(1)

    print("  ✓ Connected to ESP32.\n")

    # ── Test sequence ──────────────────────────────────────────────────────────
    tests = [
        ("STATUS",    lambda: esp.query_status()),
        # ("FORWARD",   lambda: esp.move("forward",  speed=150)),
        # ("Wait 1s",   lambda: time.sleep(1) or None),
        # ("LEFT",      lambda: esp.move("left",     speed=150)),
        # ("Wait 1s",   lambda: time.sleep(1) or None),
        # ("RIGHT",     lambda: esp.move("right",    speed=150)),
        # ("Wait 1s",   lambda: time.sleep(1) or None),
        # ("BACKWARD",  lambda: esp.move("backward", speed=150)),
        # ("Wait 1s",   lambda: time.sleep(1) or None),
        ("STOP",      lambda: esp.stop()),
        ("Wait 1s",   lambda: time.sleep(100) or None),
        ("MODE AUTO", lambda: esp.set_mode("auto")),
        ("Wait 1s",   lambda: time.sleep(1) or None),
        ("MODE MANUAL", lambda: esp.set_mode("manual")),
        ("STATUS",    lambda: esp.query_status()),
    ]

    for name, action in tests:
        if name.startswith("Wait"):
            print(f"  ⏳  {name}...")
            action()
            continue

        print(f"  ► Sending: {name}... ", end="", flush=True)
        resp = action()
        if resp is None:
            print()
            continue

        if resp.ok:
            status_info = f"  msg='{resp.msg}'" if resp.msg else ""
            sensors_info = ""
            if resp.sensors:
                s = resp.sensors
                sensors_info = (f"  | sensors: front={s.get('front')}mm "
                                f"left={s.get('left')}mm right={s.get('right')}mm")
            print(f"✓ OK{status_info}  ({resp.latency_ms:.1f}ms){sensors_info}")
        else:
            print(f"✗ FAILED — error='{resp.error}'  ({resp.latency_ms:.1f}ms)")

    print("\n  ✓ Test complete. Review results above.\n")
    esp.close()

if __name__ == "__main__":
    main()
