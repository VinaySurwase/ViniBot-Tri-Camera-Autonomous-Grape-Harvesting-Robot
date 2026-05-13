#!/usr/bin/env python3
"""
tools/test_esp32_sensors.py
============================
Reads all 3 HC-SR04 ultrasonic sensor values from the ESP32 over UART
in a live, continuously updating loop.

Use this to:
  1. Verify sensors are wired correctly
  2. Check obstacle detection distances
  3. Confirm auto-stop triggers at the right threshold

Press Ctrl+C to stop.

Usage:
  python tools/test_esp32_sensors.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml


def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def bar(value_mm: int, max_mm: int = 2000, width: int = 30) -> str:
    """Draw a simple ASCII bar for visual distance feedback."""
    if value_mm >= 9000:
        return "[" + "~" * width + "] (out of range)"
    ratio = min(value_mm / max_mm, 1.0)
    filled = int(ratio * width)
    color = "\033[92m" if value_mm > 300 else ("\033[93m" if value_mm > 150 else "\033[91m")
    reset = "\033[0m"
    bar_str = f"{color}{'█' * filled}{'░' * (width - filled)}{reset}"
    return f"[{bar_str}] {value_mm:5d} mm"


def main():
    cfg = load_cfg()
    obstacle_mm = cfg.get("esp32", {}).get("obstacle_stop_mm", 200)

    from comms.esp32_comms import ESP32Comms
    esp = ESP32Comms(cfg)

    print("\n" + "=" * 60)
    print("  ESP32 Ultrasonic Sensor Live Monitor")
    print("=" * 60)
    print(f"  Port         : {esp.port}")
    print(f"  Auto-stop at : {obstacle_mm} mm")
    print(f"  🟢 Safe  > 300 mm")
    print(f"  🟡 Warn  150–300 mm")
    print(f"  🔴 STOP  < {obstacle_mm} mm  (obstacle!)")
    print("=" * 60)
    print("  Press Ctrl+C to quit.\n")

    if not esp.connect():
        print("  ✗ Failed to connect to ESP32. Check UART wiring and port.")
        sys.exit(1)

    print("  ✓ Connected. Reading sensors every 100ms...\n")

    read_count = 0
    try:
        while True:
            resp = esp.query_status()
            read_count += 1

            if not resp.ok:
                print(f"\r  ⚠ ESP32 error: {resp.error}   ", end="")
                time.sleep(0.5)
                continue

            s = resp.sensors
            front = s.get("front", 9999)
            left  = s.get("left",  9999)
            right = s.get("right", 9999)

            # Build a multi-line display using ANSI escapes to overwrite in-place
            lines = [
                f"  Read #{read_count:04d}  |  State: {resp.status or 'unknown'}",
                f"",
                f"  FRONT  {bar(front)}  {'⛔ OBSTACLE!' if front < obstacle_mm else ''}",
                f"  LEFT   {bar(left)}",
                f"  RIGHT  {bar(right)}",
                f"",
            ]

            # Move cursor up by number of lines and overwrite
            if read_count > 1:
                sys.stdout.write(f"\033[{len(lines)}A")

            for line in lines:
                sys.stdout.write(line.ljust(70) + "\n")
            sys.stdout.flush()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n  Exiting sensor monitor.\n")
    finally:
        esp.close()


if __name__ == "__main__":
    main()
