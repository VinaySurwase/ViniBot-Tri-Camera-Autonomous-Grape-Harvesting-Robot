"""
logging_/logger.py — Structured Logger v2
==========================================
Console + rotating file logging.
Machine-readable events.csv for post-run analysis.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from colorama import Fore, Style, init as _ci
    _ci(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    class _S:
        def __getattr__(self, _): return ""
    Fore = Style = _S()
    _HAS_COLOR = False

# ── Level colours ─────────────────────────────────────────────────────────────
_LC = {"DEBUG": Fore.CYAN, "INFO": Fore.GREEN,
       "WARNING": Fore.YELLOW, "ERROR": Fore.RED, "CRITICAL": Fore.RED}


class _ColFmt(logging.Formatter):
    FMT = "[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    DATEFMT = "%H:%M:%S"
    def format(self, r):
        msg = logging.Formatter(self.FMT, self.DATEFMT).format(r)
        return f"{_LC.get(r.levelname,'')}{msg}{Style.RESET_ALL}" if _HAS_COLOR else msg


class _PlainFmt(logging.Formatter):
    FMT = "[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"
    def format(self, r):
        return logging.Formatter(self.FMT, self.DATEFMT).format(r)


# ── Module-level state ────────────────────────────────────────────────────────
_run_dir: Optional[Path]  = None
_csv_file  = None
_csv_writer = None
_run_start: float = time.time()
_handlers_set: bool = False


def init_run_dir(log_root: str = "logs", max_runs: int = 30) -> Path:
    global _run_dir, _csv_file, _csv_writer
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)
    _run_dir = run_dir

    # CSV event log
    csv_path = run_dir / "events.csv"
    _csv_file   = open(csv_path, "w", newline="")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(["ts_iso", "elapsed_s", "state", "event",
                           "value", "latency_ms", "extra"])
    _csv_file.flush()

    # Prune old runs
    runs = sorted(root.glob("run_*"))
    while len(runs) > max_runs:
        import shutil
        shutil.rmtree(runs.pop(0), ignore_errors=True)

    return run_dir


def setup_root_logger(console_level: str = "INFO",
                      file_level: str = "DEBUG") -> None:
    global _handlers_set, _run_start
    if _handlers_set:
        return
    _handlers_set = True
    _run_start    = time.time()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(_ColFmt())
    root.addHandler(ch)

    if _run_dir:
        fh = logging.FileHandler(_run_dir / "grapebot.log")
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(_PlainFmt())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_run_dir() -> Optional[Path]:
    return _run_dir


def get_frames_dir() -> Optional[Path]:
    return _run_dir / "frames" if _run_dir else None


def log_event(
    state: str,
    event: str,
    value: str = "",
    latency_ms: float = 0.0,
    extra: str = "",
) -> None:
    if _csv_writer is None:
        return
    elapsed = time.time() - _run_start
    _csv_writer.writerow([
        datetime.now().isoformat(timespec="milliseconds"),
        f"{elapsed:.3f}",
        state, event, value,
        f"{latency_ms:.2f}", extra,
    ])
    _csv_file.flush()


def close_logs() -> None:
    if _csv_file:
        _csv_file.close()