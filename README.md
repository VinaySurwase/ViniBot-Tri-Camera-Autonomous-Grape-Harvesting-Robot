# 🍇 ViniBot — Tri-Camera Autonomous Grape Harvesting Robot
### Device: phoenix (Raspberry Pi 5 — 8GB) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-c51a4a.svg)](https://www.raspberrypi.com/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n%20NCNN-00BFFF.svg)](https://docs.ultralytics.com/)

> **ViniBot** is a fully autonomous grape-harvesting robot built on a Raspberry Pi 5. It uses a tri-camera setup (dual stereo PiCamera V3 + USB cam) to detect grape clusters, compute 3D coordinates via stereo depth, and command a SCARA robotic arm to segment and cut stems — all controlled by a 7-state finite state machine.

---

## 🔗 Related Repositories

| Component | Repository | Description |
|---|---|---|
| 📱 **Android App** | [VinaySurwase/ViniBot-android](https://github.com/VinaySurwase/ViniBot-android) | Joystick control, mode switching (Manual/Auto), real-time coordinate debug panel via WebSocket |
| 🦾 **STM32 Arm Controller** | [bhoomikahardwani09/scara_robotic_arm](https://github.com/bhoomikahardwani09/scara_robotic_arm) | SCARA robotic arm firmware — receives `x#y#z` coordinates over UART and positions the end-effector |
| 🤖 **ESP32 Chassis Controller** | [`esp32/vinibot_esp32/`](esp32/vinibot_esp32/vinibot_esp32.ino) | Motor driver (BTS7960B), scissor servo, ultrasonic obstacle avoidance — included in this repo |

---

## 🔧 Hardware Requirements

| Component | Model / Spec | Qty |
|---|---|---|
| Single-board computer | Raspberry Pi 5 — 8 GB | 1 |
| Stereo cameras | Raspberry Pi Camera Module V3 | 2 |
| Segmentation camera | USB Camera (1280×720, UVC) | 1 |
| Arm controller | STM32 microcontroller (USART) | 1 |
| Chassis controller | ESP32 (UART2 via GPIO16/17) | 1 |
| Motor driver | BTS7960B dual H-bridge | 2 |
| Drive motors | 30 RPM DC gear motors | 4 |
| End-effector servo | Standard servo (0–44° range) | 1 |
| Scissor servo | Standard servo (0–80° range) | 1 |
| Ultrasonic sensors | HC-SR04 (front, left, right) | 3 |
| Power supply | 5V/5A (Pi) + 12V (motors) | — |

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    phoenix  (Raspberry Pi 5)                     │
│                                                                  │
│  Pi Cam V3 L (idx 0)    Pi Cam V3 R (idx 1)    USB Camera       │
│  via picamera2           via picamera2           via OpenCV      │
│  STEREO LEFT             STEREO RIGHT            SEGMENTATION    │
│       │                       │                       │          │
│       └──────────┬────────────┘                       │          │
│              Detection                          Segmentation     │
│         HSV + HoughCircles                   YOLOv8n-seg NCNN   │
│         YOLOv8n NCNN fallback                Cutting point       │
│              │                                        │          │
│         Stereo Depth (SGBM)               Depth @ cut point     │
│              │                                        │          │
│         POI coords (x,y,z)              CUT coords (x,y,z)      │
│              └──────────────────┬─────────────────────┘          │
│                           State Machine FSM                      │
│              IDLE → DETECT → APPROACH → SEGMENT → CUT            │
│              MANUAL ← Android joystick                           │
│                                │                                 │
│                      WebSocket :8765  ←→  ViniBot Android App   │
│                                │                                 │
│                         STM32 (serial / WiFi)                    │
│                         Bot motion + Arm control                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/home/phoenix/bot/
│
├── main.py                      ← Entry point — run this
├── setup_phoenix.sh             ← One-shot Pi 5 setup script
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
│
├── config/
│   └── config.yaml              ← ALL tunable parameters
│
├── core/
│   ├── camera_manager.py        ← PiCameraV3 (picamera2) + USBCamera (OpenCV)
│   ├── state_machine.py         ← 7-state FSM controller
│   └── false_positive_tracker.py← FP zone blacklist + stem retry logic
│
├── detection/
│   └── detector.py              ← OpenCV HSV + HoughCircles → YOLO NCNN fallback
│
├── segmentation/
│   ├── segmentor.py             ← YOLOv8n-seg NCNN on USB cam
│   └── cutting_point.py        ← Topmost / centroid / midpoint extraction
│
├── depth/
│   └── stereo_depth.py         ← SGBM calibrated depth + ORB fallback
│
├── comms/
│   ├── stm32_comms.py          ← Serial + WiFi transport to STM32
│   └── ws_server.py            ← WebSocket server (ViniBot Android app)
│
├── logging_/
│   ├── logger.py               ← Console + file + CSV event log
│   ├── frame_saver.py          ← Annotated frame saves for visual review
│   └── coord_debug.py          ← JSONL coordinate debug log
│
├── calibration/
│   └── run_calibration.py      ← Interactive stereo calibration (picamera2)
│
├── tools/
│   ├── tune_hsv.py             ← Live HSV threshold tuner
│   ├── test_subsystem.py       ← Per-module test runner
│   ├── verify_depth.py         ← Depth accuracy verifier at known distances
│   └── analyse_log.py          ← Post-run log analyser + recommendations
│
├── weights/
│   ├── det_ncnn_model/         ← YOLOv8n detection NCNN (.param + .bin)
│   └── seg_ncnn_model/         ← YOLOv8n-seg NCNN (.param + .bin)
│
└── logs/                       ← Auto-created per run
    └── run_YYYYMMDD_HHMMSS/
        ├── grapebot.log        ← Full structured log
        ├── events.csv          ← Machine-readable event log
        ├── coords.jsonl        ← Every coordinate sent (POI + CUT)
        └── frames/
            ├── det_XXXXXX.jpg  ← Annotated detection frames
            ├── seg_XXXXXX.jpg  ← Segmentation + cutting point frames
            └── depth_XXXXXX.jpg← Side-by-side stereo + disparity map
```

---

## 📦 Model Weights

NCNN model weights are **not included** in this repository (large binaries). Download them from the **[GitHub Releases](../../releases)** page and follow the instructions in [`weights/README.md`](weights/README.md).

```bash
# After downloading det_ncnn_model.zip and seg_ncnn_model.zip:
unzip det_ncnn_model.zip -d weights/det_ncnn_model/
unzip seg_ncnn_model.zip -d weights/seg_ncnn_model/
```

---

## Setup on phoenix

### Step 1 — Copy project

```bash
# From your development machine:
scp -r GrapeBot_Pi phoenix@<pi-ip>:/home/phoenix/bot

# Or clone / pull from git on the Pi:
cd /home/phoenix
git clone <your-repo-url> bot
```

### Step 2 — Run setup script

```bash
cd /home/phoenix/bot
chmod +x setup_phoenix.sh
./setup_phoenix.sh
```

This installs:
- System packages: libcamera, picamera2, OpenCV, v4l-utils, build tools
- Python virtual environment (`venv/`) with `--system-site-packages` so
  the system-installed picamera2 is accessible
- All Python packages from `requirements.txt`
- Adds `phoenix` user to `dialout` group (STM32 serial)
- Creates `weights/`, `calibration/`, `logs/` directories
- Creates a systemd service `grapebot.service` (disabled by default)
- Runs an import smoke test

### Step 3 — Copy NCNN weights

```bash
# Detection model (YOLOv8n)
scp best_ncnn_model/* phoenix@<pi-ip>:/home/phoenix/bot/weights/det_ncnn_model/

# Segmentation model (YOLOv8n-seg)
scp best_seg_ncnn_model/* phoenix@<pi-ip>:/home/phoenix/bot/weights/seg_ncnn_model/
```

### Step 4 — Stereo calibration (first time only)

```bash
source venv/bin/activate

# Print a chessboard: 9×6 inner corners, measure your square size in mm
python calibration/run_calibration.py --square 25.0

# Controls: SPACE=capture  d=delete last  c=calibrate now  q=quit
# Capture 15–20 pairs from varied angles (tilt, rotate, different distances)
# RMS error < 1.0 px = good;  < 0.5 px = excellent
```

### Step 5 — Tune HSV under vineyard lighting

```bash
# Run BEFORE going to the field — lighting changes colour appearance
python tools/tune_hsv.py                         # uses picamera2 index 0
python tools/tune_hsv.py --source usb            # USB cam instead
python tools/tune_hsv.py --image captured.jpg    # static image

# Press 's' to save to config/config.yaml automatically
```

### Step 6 — Verify depth

```bash
# Place a flat target at known distances (100mm, 200mm, 300mm, etc.)
python tools/verify_depth.py --distances 100 200 300 400 500

# Press SPACE at each distance — compares measured vs expected
# Mean error < 20mm = good;  > 60mm = re-calibrate or fix baseline_mm
```

### Step 7 — Test all subsystems

```bash
python tools/test_subsystem.py all --show

# Or individually:
python tools/test_subsystem.py cameras           # opens all 3 cameras
python tools/test_subsystem.py detection --show  # live detection overlay
python tools/test_subsystem.py seg       --show  # segmentation + cut point
python tools/test_subsystem.py depth     --show  # disparity map display
python tools/test_subsystem.py coords            # fake coord logging test
python tools/test_subsystem.py stm32    --dry-run# STM32 command test
```

### Step 8 — Start the robot

```bash
# Safe first run — no STM32 movement, full debug output
python main.py --dry-run --debug --coord-debug

# Full run
python main.py --coord-debug --save-frames

# Auto mode from startup (no need to use Android app to switch)
python main.py --mode auto --coord-debug
```

---

## CLI Flags

| Flag | Description |
|------|-------------|
| `--debug` | DEBUG-level logging to console |
| `--dry-run` | Log STM32 commands but never send them |
| `--no-depth` | Disable stereo depth computation |
| `--coord-debug` | Write every coordinate to `logs/coords.jsonl` |
| `--mode auto` | Start in auto mode immediately |
| `--mode manual` | Start in manual mode (default — wait for Android) |
| `--save-frames` | Force save annotated frames every N detections |
| `--ws-port N` | Override WebSocket port (default 8765) |
| `--config PATH` | Use alternative config file |

---

## Camera Assignment

| Camera | Type | Role | Index |
|--------|------|------|-------|
| Pi Camera V3 #1 | picamera2 | Stereo LEFT — grape detection | 0 |
| Pi Camera V3 #2 | picamera2 | Stereo RIGHT — depth + detection | 1 |
| USB Camera | OpenCV | Segmentation — stem cutting point | /dev/video0 |

**Camera exclusivity** — the state machine enforces mutual exclusion:
- `DETECT` / `APPROACH` states → only stereo V3 cams run
- `SEGMENT` / `CUT` states → only USB cam runs
- `MANUAL` state → no inference cameras active

---

## State Machine

```
IDLE
  │  Android sends start / mode=auto
  ▼
DETECT  ← stereo cams only
  │  Cluster confirmed in N consecutive frames
  ▼
APPROACH  ← send POI (x,y,z) to STM32 → bot moves
  │  Bot arrives at position
  ▼
SEGMENT  ← USB cam only
  │  Valid cutting point found + depth computed
  ▼
CUT  ← send CUT (x,y,z) to STM32 → arm cuts
  │  Cut complete
  ▼
DETECT  (next cluster)

Any state → MANUAL  (Android sends mode=manual)
Any state → ERROR   (unrecoverable failure)
ERROR     → IDLE    (Android sends reset)
```

---

## Coordinate Debug

Every coordinate sent to STM32 is written to `logs/run_XXX/coords.jsonl`:

```jsonl
{"ts": 1712345678.1, "type": "poi", "x": 320.0, "y": 240.0, "z": 450.0, "conf": 0.78, "method": "opencv", "state": "DETECT"}
{"ts": 1712345682.4, "type": "cut", "x": 215.0, "y": 180.0, "z": 430.0, "stem_idx": 0, "area_px": 412, "state": "SEGMENT"}
```

To verify coordinates are correct:
```bash
# Live (during a run)
tail -f logs/run_latest/coords.jsonl | python3 -m json.tool

# Post-run analysis
python tools/analyse_log.py
python tools/analyse_log.py --all          # compare all runs
python tools/analyse_log.py --export out.json
```

Coordinates are also pushed live to the ViniBot Android app Debug tab.

---

## False Positive Handling

| Parameter | Config key | Default | Effect |
|-----------|-----------|---------|--------|
| Cluster confirmation frames | `frames_to_confirm` | 3 | Cluster must appear in N consecutive frames |
| Min confidence | `min_cluster_confidence` | 0.55 | Below this → ignored |
| Max detection retries | `max_detection_retries` | 5 | After N failures → zone blacklisted |
| FP zone IOU threshold | `fp_iou_threshold` | 0.3 | New detection overlapping FP zone → skip |
| FP zone cooldown | hardcoded | 30s | Zone expires after 30 seconds |
| Max stem retries | `max_stem_retries` | 3 | After N seg failures → stem marked FP |
| Stem consistency | `min_stem_consistency` | 2 | Stem must appear in 2 consecutive seg frames |

---

## STM32 Protocol

```json
{"cmd": "move",   "dir": "forward|backward|left|right|stop", "speed": 0-255}
{"cmd": "coords", "type": "poi|cut", "x": f, "y": f, "z": f}
{"cmd": "status"}
{"cmd": "home"}
```

Response: `{"ok": true, "msg": "..."}` or `{"ok": false, "error": "..."}`

Transport: serial (`/dev/ttyUSB0`) or WiFi (TCP socket) — set in `config.yaml`.

---

## Run as a Service

```bash
# Enable auto-start on boot
sudo systemctl enable grapebot

# Start / stop
sudo systemctl start grapebot
sudo systemctl stop grapebot

# View live logs
journalctl -u grapebot -f

# Disable auto-start
sudo systemctl disable grapebot
```

---

## Key Config Parameters

| Parameter | Location in config.yaml | Default |
|-----------|--------------------------|---------|
| Pi Cam LEFT index | `cameras.stereo_left.index` | 0 |
| Pi Cam RIGHT index | `cameras.stereo_right.index` | 1 |
| USB cam index | `cameras.usb_seg.index` | 0 |
| HSV lower bound | `detection.hsv_lower` | [100,50,50] |
| HSV upper bound | `detection.hsv_upper` | [140,255,255] |
| Seg confidence | `segmentation.conf_thres` | 0.15 |
| Seg image size | `segmentation.imgsz` | 480 |
| Stereo baseline | `depth.baseline_mm` | 60.0 |
| STM32 transport | `stm32.transport` | serial |
| STM32 port | `stm32.serial_port` | /dev/ttyUSB0 |
| WebSocket port | `websocket.port` | 8765 |
| Coord debug | `logging.coord_debug` | true |

---

## 🤝 Acknowledgements

This project was developed at **[IIITDM Jabalpur](https://www.iiitdmj.ac.in/)**.

| Role | Contributor |
|---|---|
| Pi 5 pipeline, detection, segmentation, depth, FSM | [Vinay Surwase](https://github.com/VinaySurwase) |
| SCARA arm firmware & STM32 controller | [Bhoomika Hardwani](https://github.com/bhoomikahardwani09) |
| Android companion app | [Vinay Surwase](https://github.com/VinaySurwase) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.