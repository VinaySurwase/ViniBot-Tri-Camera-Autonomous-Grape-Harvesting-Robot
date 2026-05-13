#!/usr/bin/env python3
"""
tools/test_ws_mock.py
=====================
A simple mock script to test the connection between the Android App 
and the Raspberry Pi. It starts the WebSocket server and prints
every command the Android App sends to the terminal.
"""

import sys
import time
import logging
from pathlib import Path
import yaml

# Ensure we can import from the parent directory (bot)
sys.path.insert(0, str(Path(__file__).parent.parent))

from comms.ws_server import WSServer

# Set up logging so we can see connection messages from the server
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg = load_cfg(cfg_path)
    
    server = WSServer(cfg)
    
    # ── Callbacks that print what the Android app sends ──
    
    def on_mode_change(mode):
        print(f"\n🟢 [ANDROID INPUT] MODE CHANGE -> {mode.upper()}\n")
        
    def on_move_cmd(direction, speed):
        print(f"\n🕹️ [ANDROID INPUT] JOYSTICK MOVE -> Direction: {direction.upper()} | Speed: {speed}\n")
        
    def on_control_cmd(cmd):
        print(f"\n⚙️ [ANDROID INPUT] CONTROL COMMAND -> {cmd.upper()}\n")
        
    server.on_mode_change(on_mode_change)
    server.on_move_cmd(on_move_cmd)
    server.on_control_cmd(on_control_cmd)
    
    print("\n" + "=" * 55)
    print("  Android App Connection Tester")
    print("=" * 55)
    print(f"  Listening on {server.host}:{server.port}...")
    print("  Open your Android App and connect to the Pi's IP.")
    print("  Press Ctrl+C to stop.")
    print("=" * 55 + "\n")
    
    server.start()
    
    try:
        # Keep the main thread alive while the background WS thread listens
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()

if __name__ == "__main__":
    main()