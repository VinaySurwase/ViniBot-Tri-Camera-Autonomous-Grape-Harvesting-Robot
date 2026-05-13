#!/usr/bin/env python3
"""
tools/test_speeds.py
====================
Interactive tool to test ESP32 chassis movement at different PWM speeds.
Allows you to verify motor start thresholds and high-speed stability.

Controls:
  w/s — Forward / Backward
  a/d — Left / Right (turn)
  x   — Stop
  +/- — Increase / Decrease speed by 10
  0-9 — Set speed presets (1=30, 2=50, 3=80, ..., 9=255, 0=Stop)
  q   — Quit

Usage:
  python tools/test_speeds.py
  python tools/test_speeds.py --dry-run
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
try:
    import getch
except ImportError:
    # Simple fallback if getch is not installed
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
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simulate without hardware")
    parser.add_argument("--ramp", action="store_true", help="Automatically ramp speed 0->255")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.dry_run:
        cfg.setdefault("esp32", {})["dry_run"] = True

    from comms.esp32_comms import ESP32Comms
    esp = ESP32Comms(cfg)

    print("\n" + "=" * 60)
    print("  ESP32 Speed & Movement Tester")
    print("=" * 60)
    
    if args.ramp:
        print("  MODE: Automatic Speed Ramp")
        print("=" * 60)
    else:
        print("  Controls:")
        print("    W/S : Forward/Backward    A/D : Left/Right")
        print("    X   : Stop                Q   : Quit")
        print("    +/- : Speed +/- 10        0-9 : Presets")
        print("=" * 60)

    if not esp.connect():
        print("  ✗ Failed to connect to ESP32.")
        sys.exit(1)

    esp.set_mode("manual")
    
    if args.ramp:
        try:
            print("\n  Ramping speed forward 0 -> 255 in steps of 25...")
            for s in range(0, 256, 25):
                print(f"\r  Testing speed: {s} ", end="", flush=True)
                esp.move("forward", s)
                time.sleep(1.5)
            print("\n  Ramping speed backward 0 -> 255 in steps of 25...")
            for s in range(0, 256, 25):
                print(f"\r  Testing speed: {s} ", end="", flush=True)
                esp.move("backward", s)
                time.sleep(1.5)
            print("\n  Stopping.")
            esp.stop()
            print("  ✓ Ramp complete.")
        except KeyboardInterrupt:
            print("\n\n  Interrupted.")
            esp.stop()
        finally:
            esp.close()
            return

    current_speed = 50
    steering_speed = 30
    last_dir = "stop"

    def send_move(direction, speed):
        nonlocal last_dir
        last_dir = direction
        if direction == "stop":
            resp = esp.stop()
        else:
            resp = esp.move(direction, speed)
        
        if resp.ok:
            print(f"\r  ► {direction.upper():<8} | Speed: {speed:>3} | OK ({resp.latency_ms:.1f}ms)      ", end="", flush=True)
        else:
            print(f"\r  ✗ {direction.upper():<8} | FAILED: {resp.error}      ", end="", flush=True)

    print(f"\n  Initial Speed: {current_speed}")
    print("  Ready. Press a key to move...\n")

    try:
        while True:
            char = getch().lower()

            if char == 'q':
                print("\n\n  Quitting...")
                esp.stop()
                break
            
            elif char == 'w': send_move("forward", current_speed)
            elif char == 's': send_move("backward", current_speed)
            elif char == 'a': send_move("left", steering_speed)
            elif char == 'd': send_move("right", steering_speed)
            elif char == 'x': send_move("stop", 0)
            
            elif char == '+':
                current_speed = min(255, current_speed + 10)
                if last_dir != "stop": send_move(last_dir, current_speed)
                else: print(f"\r  Speed set to: {current_speed}            ", end="", flush=True)
            
            elif char == '-':
                current_speed = max(0, current_speed - 10)
                if last_dir != "stop": send_move(last_dir, current_speed)
                else: print(f"\r  Speed set to: {current_speed}            ", end="", flush=True)
            
            elif char.isdigit():
                val = int(char)
                if val == 0:
                    current_speed = 0
                    send_move("stop", 0)
                else:
                    # Presets: 1=30, 2=50, 3=75, 4=100, 5=125, 6=150, 7=175, 8=200, 9=255
                    presets = [0, 30, 50, 75, 100, 125, 150, 175, 200, 255]
                    current_speed = presets[val]
                    if last_dir != "stop": send_move(last_dir, current_speed)
                    else: print(f"\r  Speed set to: {current_speed}            ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        esp.stop()
    finally:
        esp.close()
        print("\n  Closed.\n")

if __name__ == "__main__":
    main()
