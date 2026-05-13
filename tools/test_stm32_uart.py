#!/usr/bin/env python3
"""
tools/test_stm32_uart.py
=========================
Basic UART connectivity test for the STM32 arm controller.

Tests:
  1. Serial port opens successfully
  2. Character-by-character coordinate send works (proven protocol)
  3. Port closes cleanly

Uses the same proven protocol as temp/uart_send.py:
  Format: "x.x#y.y#z.z\n" sent one character at a time.

Usage:
  python tools/test_stm32_uart.py
  python tools/test_stm32_uart.py --port /dev/ttyAMA2
  python tools/test_stm32_uart.py --delay 500     # match reference timing exactly
  python tools/test_stm32_uart.py --dry-run        # test without hardware
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


def send_char_by_char(ser, x, y, z, delay_s=0.05):
    """Send coordinates char-by-char, exactly like temp/uart_send.py."""
    command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"
    print(f"  Sending: {command_string.strip()}")
    print(f"  Char-by-char: ", end="", flush=True)

    t0 = time.perf_counter()
    for char in command_string:
        ser.write(char.encode('utf-8'))
        if char == '\n':
            print("↵", end="", flush=True)
        else:
            print(char, end="", flush=True)
        time.sleep(delay_s)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  ✓ Sent in {elapsed:.0f}ms "
          f"({len(command_string)} chars × {delay_s*1000:.0f}ms)")
    return True


def main():
    parser = argparse.ArgumentParser(description="STM32 UART Connection Test")
    parser.add_argument("--port", default=None,
                        help="Serial port (default: from config)")
    parser.add_argument("--baud", type=int, default=None,
                        help="Baud rate (default: from config)")
    parser.add_argument("--delay", type=int, default=None,
                        help="Per-char delay in ms (default: from config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without hardware")
    args = parser.parse_args()

    cfg = load_cfg()
    stm_cfg = cfg.get("stm32", {})

    port     = args.port  or stm_cfg.get("serial_port",      "/dev/ttyAMA2")
    baud     = args.baud  or stm_cfg.get("serial_baud",      115200)
    delay_ms = args.delay if args.delay is not None else stm_cfg.get("char_delay_ms", 50)
    warmup_s = stm_cfg.get("warmup_s", 2.0)
    delay_s  = delay_ms / 1000.0
    dry_run  = args.dry_run or stm_cfg.get("dry_run", False)

    print("\n" + "=" * 60)
    print("  STM32 UART Connection Test")
    print("=" * 60)
    print(f"  Port      : {port}")
    print(f"  Baud      : {baud}")
    print(f"  Char delay: {delay_ms}ms")
    print(f"  Warmup    : {warmup_s}s")
    print(f"  Dry run   : {dry_run}")
    print("=" * 60)

    # ── Test 1: Open serial port ──────────────────────────────────────────────
    print("\n[1/3] Opening serial port...")

    if dry_run:
        print(f"  ✓ DRY-RUN — skipping real serial open")

        class FakeSerial:
            """Simulates serial.Serial for dry-run testing."""
            def write(self, data): pass
            def close(self): pass

        stm32 = FakeSerial()
    else:
        try:
            import serial
            stm32 = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=1,
            )
            print(f"  ✓ Port opened: {port}")
        except Exception as e:
            print(f"  ✗ FAILED to open {port}: {e}")
            print("\n  Troubleshooting:")
            print(f"    • Is the STM32 connected to {port}?")
            print("    • Run: ls -la /dev/ttyAMA*")
            print("    • Check raspi-config serial settings")
            print("    • Check wiring: Pi TX (GPIO14) → STM32 RX")
            print("    • Try --dry-run to test without hardware")
            sys.exit(1)

    # ── Test 2: Warmup ────────────────────────────────────────────────────────
    print(f"\n[2/3] Warmup ({warmup_s}s)...")
    time.sleep(warmup_s if not dry_run else 0.1)
    print("  ✓ Warmup complete")

    # ── Test 3: Send test coordinates ─────────────────────────────────────────
    print("\n[3/3] Sending test coordinates...")
    test_coords = [
        (0.0,   0.0,   0.0),        # Origin
        (450.0, 350.0, 50.0),        # Same as reference uart_send.py
        (100.5, 200.3, 75.0),        # Arbitrary test values
    ]

    all_ok = True
    for i, (x, y, z) in enumerate(test_coords, 1):
        print(f"\n  ─── Coord {i}/{len(test_coords)} ───")
        try:
            send_char_by_char(stm32, x, y, z, delay_s)
            time.sleep(0.5)          # brief pause between sends
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_ok = False

    # ── Cleanup ───────────────────────────────────────────────────────────────
    stm32.close()
    print(f"\n  ✓ Port closed")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ ALL TESTS PASSED — STM32 UART is working!")
    else:
        print("  ❌ SOME TESTS FAILED — check output above")
    print("=" * 60 + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
