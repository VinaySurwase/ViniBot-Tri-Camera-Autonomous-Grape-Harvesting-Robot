#!/usr/bin/env python3
"""
tools/test_uart_isolated.py
Isolated script to test sending X, Y, Z coordinates to STM32 via UART.
Does NOT import any project dependencies (no config, no comms, etc.).
Requires only the standard `pyserial` library.
"""

import serial
import time
import argparse
import sys

def send_char_by_char(ser, x, y, z, delay_s):
    """
    Format: "x.x#y.y#z.z\n"
    Sends one character at a time with a delay to prevent buffer overruns.
    """
    command_string = f"{x:.1f}#{y:.1f}#{z:.1f}\n"
    print(f"  TX: ", end="", flush=True)

    t0 = time.perf_counter()
    for char in command_string:
        ser.write(char.encode('utf-8'))
        print(char if char != '\n' else '↵', end="", flush=True)
        time.sleep(delay_s)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  ({elapsed:.0f}ms)")

def main():
    parser = argparse.ArgumentParser(description="Isolated STM32 UART Coordinate Sender")
    parser.add_argument("--port", type=str, default="/dev/ttyAMA2", help="Serial port (default: /dev/ttyAMA2)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--delay", type=int, default=50, help="Delay between characters in ms (default: 50)")
    parser.add_argument("-x", type=float, default=90.0, help="X coordinate (default: 90.0)")
    parser.add_argument("-y", type=float, default=250.0, help="Y coordinate/depth (default: 250.0)")
    parser.add_argument("-z", type=float, default=325.0, help="Z coordinate/height (default: 325.0)")
    parser.add_argument("--loop", action="store_true", help="Run interactive loop to manually type coords")
    
    args = parser.parse_args()

    print("="*50)
    print(" STM32 UART Isolated Tester")
    print("="*50)
    print(f" Port      : {args.port}")
    print(f" Baud      : {args.baud}")
    print(f" Char delay: {args.delay} ms")
    print("="*50)

    try:
        ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=1)
    except Exception as e:
        print(f"\n[ERROR] Could not open port {args.port}: {e}")
        print("Check your wiring, check 'ls -l /dev/ttyAMA*', or run with sudo.")
        sys.exit(1)

    print("\nWarming up connection (2s)...")
    time.sleep(2.0)
    ser.reset_input_buffer()

    delay_s = args.delay / 1000.0

    if args.loop:
        print("\nInteractive Mode: Type 3 numbers separated by space (e.g. '90 300 250')")
        print("Type 'q' to quit.")
        try:
            while True:
                user_input = input("\nX Y Z > ").strip()
                if user_input.lower() == 'q':
                    break
                
                parts = user_input.split()
                if len(parts) == 3:
                    try:
                        x, y, z = map(float, parts)
                        send_char_by_char(ser, x, y, z, delay_s)
                    except ValueError:
                        print(" Invalid numbers. Please enter digits.")
                else:
                    print(" Please enter exactly three numbers (X Y Z).")
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        print(f"\nSending single command:")
        send_char_by_char(ser, args.x, args.y, args.z, delay_s)

    ser.close()
    print("\nPort closed. Done.")

if __name__ == "__main__":
    main()
