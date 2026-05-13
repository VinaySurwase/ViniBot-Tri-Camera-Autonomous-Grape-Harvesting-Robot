#!/usr/bin/env python3
"""
tools/monitor_android.py
========================
Provides a live feed of all Android App commands being received 
by the background `vinibot.service`.
Does NOT start a new server. Just taps into the live log file.
"""

import os
import glob
import time

def main():
    logs_dir = "/home/phoenix/bot/logs"
    
    # 1. Find the most recent run folder created by vinibot.service
    runs = glob.glob(f"{logs_dir}/run_*")
    if not runs:
        print(f"No logs found in {logs_dir}!")
        print("Make sure vinibot.service is currently running.")
        return
    
    latest_run = max(runs, key=os.path.getmtime)
    log_file = os.path.join(latest_run, "grapebot.log")

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    print("\n" + "=" * 55)
    print("  Live Android App Monitor (via vinibot.service)")
    print("=" * 55)
    print(f"  Tapping into: {log_file}")
    print("  Press Ctrl+C to exit.\n")

    try:
        # 2. Open the file and jump straight to the bottom
        with open(log_file, "r") as f:
            f.seek(0, os.SEEK_END)
            
            # 3. Continuously tail the file (like `tail -f`)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                # 4. Only print lines that involve the Android app!
                if "Android" in line:
                    # Make it green so it's easy to read
                    print(f"\033[92m{line.strip()}\033[0m")

    except KeyboardInterrupt:
        print("\nExiting monitor...")

if __name__ == "__main__":
    main()
