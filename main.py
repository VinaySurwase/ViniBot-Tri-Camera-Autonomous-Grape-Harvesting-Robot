#!/usr/bin/env python3
"""
main.py — ViniBot v2 Entry Point
===================================
/home/phoenix/bot/main.py

Usage:
    python main.py
    python main.py --debug
    python main.py --dry-run --no-depth
    python main.py --mode auto
    python main.py --coord-debug      # extra coordinate logging
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="GrapeBot v2 — Tri-Camera Grape Plucking Robot")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--debug",       action="store_true",  help="DEBUG logging to console")
    p.add_argument("--dry-run",     action="store_true",  help="No STM32 movement commands")
    p.add_argument("--no-depth",    action="store_true",  help="Disable stereo depth")
    p.add_argument("--no-seg",      action="store_true",  help="Skip segmentation (detect only)")
    p.add_argument("--coord-debug", action="store_true",  help="Enable coordinate debug JSONL log")
    p.add_argument("--mode",        default="manual",     choices=["auto", "manual"],
                   help="Starting mode (default: manual — wait for Android)")
    p.add_argument("--save-frames", action="store_true",  help="Force save annotated frames")
    p.add_argument("--ws-port",     type=int, default=None,
                   help="Override WebSocket port")
    return p.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    if args.debug:
        cfg["logging"]["console_level"] = "DEBUG"
    if args.dry_run:
        cfg["stm32"]["dry_run"] = True
    if args.no_depth:
        cfg["depth"]["enabled"] = False
    if args.coord_debug:
        cfg["logging"]["coord_debug"] = True
    if args.save_frames:
        cfg["logging"]["save_frames"] = True
    if args.ws_port:
        cfg["websocket"]["port"] = args.ws_port
    cfg["state_machine"]["auto_mode"] = (args.mode == "auto")
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    # ── Logging ───────────────────────────────────────────────────────────────
    from logging_.logger import init_run_dir, setup_root_logger, get_logger, log_event

    lg      = cfg["logging"]
    run_dir = init_run_dir(lg.get("log_dir", "logs"), lg.get("max_log_runs", 30))
    setup_root_logger(lg.get("console_level", "INFO"), lg.get("file_level", "DEBUG"))
    log = get_logger("main")

    log.info("=" * 64)
    log.info("  ViniBot v2")
    log.info("  Device: phoenix")
    log.info("=" * 64)
    log.info(f"  Run dir    : {run_dir}")
    log.info(f"  Config     : {args.config}")
    log.info(f"  Mode       : {args.mode}")
    log.info(f"  Dry run    : {args.dry_run}")
    log.info(f"  Depth      : {not args.no_depth}")
    log.info(f"  Coord debug: {cfg['logging']['coord_debug']}")
    log.info(f"  WS port    : {cfg['websocket']['port']}")
    log_event("INIT", "startup", str(run_dir))

    # Print connection info for Android app
    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        log.info(f"\n  ┌─────────────────────────────────────────┐")
        log.info(f"  │  Android: connect to ws://{ip}:{cfg['websocket']['port']}  │")
        log.info(f"  └─────────────────────────────────────────┘\n")
    except Exception:
        pass

    # ── State machine ─────────────────────────────────────────────────────────
    from core.state_machine import GrapeBotFSM

    fsm = GrapeBotFSM(cfg)

    def _shutdown(sig, frame):
        log.info("Interrupt received — shutting down.")
        fsm.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    fsm.start()

    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()