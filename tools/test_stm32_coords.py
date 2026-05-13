#!/usr/bin/env python3
"""
tools/test_stm32_coords.py
============================
Interactive tool to send custom coordinates to the STM32 arm controller.

Send coordinates manually to test arm positioning, fine-tune values,
and verify the STM32 receives them correctly.

Uses the proven char-by-char protocol from temp/uart_send.py:
  Format: "x.x#y.y#z.z\n"

Usage:
  python tools/test_stm32_coords.py                       # interactive mode
  python tools/test_stm32_coords.py --coords 450,350,50   # single-shot
  python tools/test_stm32_coords.py --delay 500            # match reference timing
  python tools/test_stm32_coords.py --dry-run              # simulate

Press Ctrl+C to quit interactive mode.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml


def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Preset coordinates for quick testing ──────────────────────────────────────
PRESETS = {
    "origin":  (0.0,   0.0,   0.0),
    "center":  (320.0, 240.0, 100.0),
    "test":    (450.0, 350.0, 50.0),       # same as temp/uart_send.py
    "far":     (600.0, 400.0, 200.0),
    "near":    (150.0, 100.0, 30.0),
    "max":     (999.9, 999.9, 999.9),
}


def send_char_by_char(ser, x, y, z, delay_s, verbose=True):
    """Send coordinates char-by-char, exactly like temp/uart_send.py."""
    command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"

    if verbose:
        print(f"  TX: ", end="", flush=True)

    t0 = time.perf_counter()
    for char in command_string:
        ser.write(char.encode('utf-8'))
        if verbose:
            print(char if char != '\n' else '↵', end="", flush=True)
        time.sleep(delay_s)

    elapsed = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  ({elapsed:.0f}ms)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="STM32 Interactive Coordinate Sender")
    parser.add_argument("--port", default=None,
                        help="Serial port override")
    parser.add_argument("--baud", type=int, default=None,
                        help="Baud rate override")
    parser.add_argument("--delay", type=int, default=None,
                        help="Per-char delay in ms (default: from config)")
    parser.add_argument("--coords", default=None,
                        help="Send single coord and exit: x,y,z  e.g. 450,350,50")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without hardware")
    
    # Pre-process sys.argv to handle negative coordinates like --coords -450,350,50
    # argparse normally treats words starting with '-' as options.
    processed_args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--coords' and i + 1 < len(sys.argv):
            next_arg = sys.argv[i+1]
            if next_arg.startswith('-') and not next_arg.startswith('--'):
                processed_args.append(f"--coords={next_arg}")
                i += 2
                continue
        processed_args.append(arg)
        i += 1

    args = parser.parse_args(processed_args)

    cfg = load_cfg()
    stm_cfg = cfg.get("stm32", {})

    port     = args.port  or stm_cfg.get("serial_port",      "/dev/ttyAMA2")
    baud     = args.baud  or stm_cfg.get("serial_baud",      115200)
    delay_ms = args.delay if args.delay is not None else stm_cfg.get("char_delay_ms", 50)
    warmup_s = stm_cfg.get("warmup_s", 2.0)
    delay_s  = delay_ms / 1000.0
    dry_run  = args.dry_run or stm_cfg.get("dry_run", False)

    print("\n" + "=" * 60)
    print("  STM32 Interactive Coordinate Sender")
    print("=" * 60)
    print(f"  Port      : {port}")
    print(f"  Baud      : {baud}")
    print(f"  Char delay: {delay_ms}ms")
    print(f"  Protocol  : x.x#y.y#z.z (char-by-char)")
    print(f"  Dry run   : {dry_run}")
    print("=" * 60)

    # ── Connect ───────────────────────────────────────────────────────────────
    if dry_run:
        class FakeSerial:
            def write(self, data): pass
            def close(self): pass
        stm32 = FakeSerial()
        print(f"  ✓ DRY-RUN mode — no data sent to hardware")
    else:
        try:
            import serial
            stm32 = serial.Serial(port=port, baudrate=baud, timeout=1)
            print(f"  ✓ Connected to {port}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print("    Try --dry-run to test without hardware")
            sys.exit(1)

    print(f"  Warming up ({warmup_s}s)...")
    time.sleep(warmup_s if not dry_run else 0.1)
    print("  ✓ Ready!\n")

    # ── Single-shot mode ──────────────────────────────────────────────────────
    if args.coords:
        try:
            parts = [float(v) for v in args.coords.split(",")]
            if len(parts) != 3:
                raise ValueError("Need exactly 3 values: x,y,z")
            x, y, z = parts
            print(f"  Sending: ({x:.1f}, {y:.1f}, {z:.1f})")
            elapsed = send_char_by_char(stm32, x, y, z, delay_s)
            print(f"  ✓ Done in {elapsed:.0f}ms\n")
        except ValueError as e:
            print(f"  ✗ Error: {e}")
        finally:
            time.sleep(0.5)
            stm32.close()
            print("  ✓ Port closed.")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  Commands:                                  │")
    print("  │    x,y,z       — send coordinates           │")
    print("  │    preset NAME — send a preset              │")
    print("  │    list        — show all presets            │")
    print("  │    repeat N    — resend last coord N times   │")
    print("  │    quit        — exit                        │")
    print("  └─────────────────────────────────────────────┘\n")

    send_count = 0
    last_coords = None

    try:
        while True:
            try:
                inp = input("  coord> ").strip()
            except EOFError:
                break

            if not inp:
                continue

            if inp.lower() in ("quit", "exit", "q"):
                break

            # ── List presets ──
            if inp.lower() == "list":
                print("  ┌─────────────────────────────────────────┐")
                print("  │  Presets:                                │")
                for name, (px, py, pz) in PRESETS.items():
                    line = f"    {name:8s} → {px:.1f}#{py:.1f}#{pz:.1f}"
                    print(f"  │{line:42s}│")
                print("  └─────────────────────────────────────────┘")
                continue

            # ── Repeat last ──
            if inp.lower().startswith("repeat"):
                if last_coords is None:
                    print("  ✗ No previous coordinates to repeat")
                    continue
                try:
                    n = int(inp.split()[1]) if len(inp.split()) > 1 else 3
                except ValueError:
                    n = 3
                x, y, z = last_coords
                print(f"\n  Repeating ({x:.1f}, {y:.1f}, {z:.1f}) × {n}")
                for i in range(n):
                    send_count += 1
                    print(f"  [#{send_count}] ", end="")
                    send_char_by_char(stm32, x, y, z, delay_s, verbose=True)
                    time.sleep(0.3)
                print(f"  ✓ Done\n")
                continue

            # ── Preset ──
            if inp.lower().startswith("preset "):
                name = inp.split(None, 1)[1].strip().lower()
                if name not in PRESETS:
                    print(f"  ✗ Unknown preset: {name}")
                    print(f"    Available: {', '.join(PRESETS)}")
                    continue
                x, y, z = PRESETS[name]

            # ── Custom coordinates ──
            else:
                try:
                    parts = [float(v) for v in inp.replace(" ", ",").split(",")]
                    if len(parts) != 3:
                        print("  ✗ Enter 3 values: x,y,z  (e.g. 450,350,50)")
                        continue
                    x, y, z = parts
                except ValueError:
                    print("  ✗ Invalid input. Try: x,y,z  or  preset NAME")
                    continue

            # ── Send ──
            send_count += 1
            last_coords = (x, y, z)
            print(f"\n  [#{send_count}] Sending ({x:.1f}, {y:.1f}, {z:.1f})")
            elapsed = send_char_by_char(stm32, x, y, z, delay_s)
            print(f"  ✓ Done\n")
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print(f"\n  Total sends: {send_count}")
    stm32.close()
    print("  ✓ Port closed. Goodbye!\n")


if __name__ == "__main__":
    main()
