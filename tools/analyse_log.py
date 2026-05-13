#!/usr/bin/env python3
"""
tools/analyse_log.py — Post-Run Log Analyser
=============================================
Parses a GrapeBot run's events.csv and grapebot.log to produce a
human-readable performance summary and calibration recommendations.

Usage:
    python tools/analyse_log.py                           # most recent run
    python tools/analyse_log.py logs/run_20250101_1430    # specific run dir
    python tools/analyse_log.py --all                     # compare all runs
    python tools/analyse_log.py --export summary.json     # save JSON report
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Colours ───────────────────────────────────────────────────────────────────

GRN = "\033[32m"; YLW = "\033[33m"; RED = "\033[31m"
CYN = "\033[36m"; DIM = "\033[2m";  RST = "\033[0m"
BOLD = "\033[1m"

def _g(m): return f"{GRN}{m}{RST}"
def _y(m): return f"{YLW}{m}{RST}"
def _r(m): return f"{RED}{m}{RST}"
def _c(m): return f"{CYN}{m}{RST}"
def _b(m): return f"{BOLD}{m}{RST}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GrapeBot run log analyser.")
    p.add_argument("run_dir",  nargs="?", default=None,
                   help="Run directory to analyse (default: most recent)")
    p.add_argument("--log-dir", default="logs", help="Root logs directory")
    p.add_argument("--all",    action="store_true", help="Compare all runs")
    p.add_argument("--export", default=None, help="Save JSON report to file")
    p.add_argument("--verbose",action="store_true", help="Show per-event details")
    return p.parse_args()


def latest_run(log_dir: str = "logs") -> Path:
    runs = sorted(Path(log_dir).glob("run_*"))
    if not runs:
        print(f"No runs found in {log_dir}/")
        sys.exit(1)
    return runs[-1]


def load_events(run_dir: Path) -> list:
    csv_path = run_dir / "events.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _avg(lst):       return sum(lst) / len(lst) if lst else 0.0
def _pct(lst, p):
    if not lst: return 0.0
    s = sorted(lst)
    return s[int(len(s) * p / 100)]


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_run(run_dir: Path) -> dict:
    events = load_events(run_dir)
    if not events:
        return {}

    def _by(event_name):
        return [e for e in events if e["event"] == event_name]

    detections   = _by("opencv_detection")
    yolo_hits    = _by("yolo_fallback_detection")
    confirmed    = _by("cluster_confirmed")
    seg_events   = _by("segmentation")
    cut_pts      = _by("cutting_points_extracted")
    depths       = _by("depth_computed")
    poi_sent     = _by("poi_coords_sent")
    cut_sent     = _by("cut_coords_sent")
    transitions  = _by("state_transition")
    fp_zones     = _by("fp_zone_added")
    stem_fp      = _by("stem_fp")
    frames_saved = _by("frame_saved_det") + _by("frame_saved_seg")
    errors       = [e for e in events if e.get("state") == "ERROR"]

    def _lats(evts):
        out = []
        for e in evts:
            try: out.append(float(e["latency_ms"]))
            except: pass
        return out

    def _vals(evts):
        out = []
        for e in evts:
            try: out.append(float(e["value"]))
            except: pass
        return out

    det_lats   = _lats(detections)
    seg_lats   = _lats(seg_events)
    dep_lats   = _lats(depths)
    depth_vals = _vals(depths)
    stem_counts= _vals(seg_events)

    methods = defaultdict(int)
    for e in confirmed:
        m = re.search(r"method=(\w+)", e.get("extra",""))
        if m: methods[m.group(1)] += 1

    try:    elapsed_s = float(events[-1]["elapsed_s"])
    except: elapsed_s = 0.0

    frame_files = list((run_dir / "frames").glob("*.jpg")) if (run_dir / "frames").exists() else []

    return {
        "run_dir"          : str(run_dir),
        "elapsed_s"        : elapsed_s,
        "frames_saved"     : len(frame_files),
        "det_count"        : len(detections),
        "yolo_fallbacks"   : len(yolo_hits),
        "clusters_confirmed": len(confirmed),
        "detection_methods": dict(methods),
        "det_avg_ms"       : _avg(det_lats),
        "det_p95_ms"       : _pct(det_lats, 95),
        "det_fps"          : 1000/_avg(det_lats) if _avg(det_lats) else 0,
        "seg_count"        : len(seg_events),
        "seg_avg_ms"       : _avg(seg_lats),
        "seg_p95_ms"       : _pct(seg_lats, 95),
        "seg_fps"          : 1000/_avg(seg_lats) if _avg(seg_lats) else 0,
        "avg_stems"        : _avg(stem_counts),
        "cut_points_found" : len(cut_pts),
        "poi_sent"         : len(poi_sent),
        "cut_sent"         : len(cut_sent),
        "depth_readings"   : len(depth_vals),
        "depth_avg_mm"     : _avg(depth_vals),
        "depth_min_mm"     : min(depth_vals) if depth_vals else 0,
        "depth_max_mm"     : max(depth_vals) if depth_vals else 0,
        "dep_avg_ms"       : _avg(dep_lats),
        "fp_zones_added"   : len(fp_zones),
        "stem_fp_events"   : len(stem_fp),
        "state_transitions": len(transitions),
        "errors"           : len(errors),
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(info: dict, verbose: bool = False) -> None:
    SEP = "─" * 64

    print(f"\n{_c('═' * 64)}")
    print(f"{_c(_b('  GRAPEBOT RUN ANALYSIS'))}")
    print(f"{_c('═' * 64)}")
    print(f"  Run dir  : {info['run_dir']}")
    print(f"  Duration : {info['elapsed_s']:.1f}s  ({info['elapsed_s']/60:.1f} min)")
    print(f"  Frames   : {info['frames_saved']} saved")

    # Detection
    print(f"\n  {_b('DETECTION  ·  Stereo V3 Cams')}")
    print(f"  {SEP}")
    print(f"    Frames processed    : {info['det_count']:>8}")
    print(f"    Clusters confirmed  : {info['clusters_confirmed']:>8}")
    print(f"    YOLO fallbacks used : {info['yolo_fallbacks']:>8}")
    print(f"    Detection methods   : {info['detection_methods'] or {'none': 0}}")

    det_color = _g if info['det_avg_ms'] < 80 else _y if info['det_avg_ms'] < 150 else _r
    if info['det_avg_ms']:
        print(f"    Avg latency         : {det_color(f\"{info['det_avg_ms']:.1f} ms\")}  "
              f"({info['det_fps']:.1f} FPS)")
        print(f"    p95 latency         : {info['det_p95_ms']:.1f} ms")

    # Segmentation
    print(f"\n  {_b('SEGMENTATION  ·  USB Cam + NCNN')}")
    print(f"  {SEP}")
    print(f"    Seg runs            : {info['seg_count']:>8}")
    print(f"    Avg stems / frame   : {info['avg_stems']:>8.1f}")
    print(f"    Cut points found    : {info['cut_points_found']:>8}")
    print(f"    POI coords sent     : {info['poi_sent']:>8}")
    print(f"    Cut coords sent     : {info['cut_sent']:>8}")

    seg_color = _g if info['seg_avg_ms'] < 150 else _y if info['seg_avg_ms'] < 300 else _r
    if info['seg_avg_ms']:
        print(f"    Avg latency         : {seg_color(f\"{info['seg_avg_ms']:.1f} ms\")}  "
              f"({info['seg_fps']:.1f} FPS)")
        print(f"    p95 latency         : {info['seg_p95_ms']:.1f} ms")

    # Depth
    print(f"\n  {_b('STEREO DEPTH')}")
    print(f"  {SEP}")
    if info['depth_readings']:
        print(f"    Depth readings      : {info['depth_readings']:>8}")
        print(f"    Avg depth           : {info['depth_avg_mm']:>8.1f} mm")
        print(f"    Range               : {info['depth_min_mm']:.1f} – {info['depth_max_mm']:.1f} mm")
        print(f"    Avg latency         : {info['dep_avg_ms']:.1f} ms")
    else:
        print(f"    {_y('No depth readings recorded.')}")

    # FP / Reliability
    print(f"\n  {_b('RELIABILITY')}")
    print(f"  {SEP}")
    fp_color = _g if info['fp_zones_added'] == 0 else _y if info['fp_zones_added'] < 3 else _r
    print(f"    FP zones added      : {fp_color(str(info['fp_zones_added'])):>8}")
    print(f"    Stem FP events      : {info['stem_fp_events']:>8}")
    print(f"    State transitions   : {info['state_transitions']:>8}")
    err_color = _g if info['errors'] == 0 else _r
    print(f"    Errors              : {err_color(str(info['errors'])):>8}")

    # Recommendations
    print(f"\n  {_b('CALIBRATION RECOMMENDATIONS')}")
    print(f"  {SEP}")
    recs = []

    yolo_rate = info['yolo_fallbacks'] / max(info['clusters_confirmed'], 1)
    if yolo_rate > 0.5:
        recs.append((_y("⚠"), "High YOLO fallback rate — run tools/tune_hsv.py to improve HSV thresholds."))

    if info['det_avg_ms'] > 100:
        recs.append((_y("⚠"), f"Detection avg {info['det_avg_ms']:.0f}ms — consider reducing "
                               "yolo_imgsz or raising hough_param2 in config.yaml."))

    if info['seg_avg_ms'] > 200:
        recs.append((_y("⚠"), f"Segmentation avg {info['seg_avg_ms']:.0f}ms — reduce imgsz in config.yaml."))

    if info['avg_stems'] == 0 and info['seg_count'] > 0:
        recs.append((_r("✗"), "Zero stems found — check seg model path and lower conf_thres."))

    if info['cut_points_found'] > 0 and info['depth_readings'] == 0:
        recs.append((_y("⚠"), "Cut points found but no depth — run calibration/run_calibration.py."))

    if info['depth_avg_mm'] and (info['depth_avg_mm'] < 80 or info['depth_avg_mm'] > 1500):
        recs.append((_y("⚠"), f"Avg depth {info['depth_avg_mm']:.0f}mm is outside 80–1500mm — "
                               "verify baseline_mm and focal_length_px in config.yaml."))

    if info['fp_zones_added'] > 5:
        recs.append((_y("⚠"), "Many FP zones — check lighting / HSV thresholds / seg model confidence."))

    if not recs:
        print(f"    {_g('✓  No issues detected — system operating normally.')}")
    else:
        for icon, msg in recs:
            print(f"    {icon}  {msg}")

    print(f"\n{_c('═' * 64)}\n")


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_all(log_dir: str = "logs") -> None:
    runs = sorted(Path(log_dir).glob("run_*"))
    if not runs:
        print("No runs found.")
        return

    print(f"\n{_c('='*76)}")
    print(f"  {_b('ALL RUNS COMPARISON')}")
    print(f"{_c('='*76)}")
    print(f"  {'Run':<26} {'Dur(s)':>6} {'Conf':>5} {'Cut':>4} "
          f"{'DetMs':>6} {'SegMs':>6} {'Depth':>6} {'FP':>3} {'Err':>3}")
    print(f"  {'─'*26} {'─'*6} {'─'*5} {'─'*4} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*3} {'─'*3}")

    for run in runs:
        info = analyse_run(run)
        if not info: continue
        err_col = _r if info['errors'] else _g
        print(f"  {run.name:<26} {info['elapsed_s']:>6.0f} "
              f"{info['clusters_confirmed']:>5} {info['cut_sent']:>4} "
              f"{info['det_avg_ms']:>6.1f} {info['seg_avg_ms']:>6.1f} "
              f"{info['depth_avg_mm']:>6.0f} "
              f"{info['fp_zones_added']:>3} "
              f"{err_col(str(info['errors'])):>3}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.all:
        compare_all(args.log_dir)
        return

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(args.log_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    info = analyse_run(run_dir)
    if not info:
        print(f"No events.csv found in {run_dir}")
        sys.exit(1)

    print_report(info, verbose=args.verbose)

    if args.export:
        with open(args.export, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  JSON report saved → {args.export}")


if __name__ == "__main__":
    main()