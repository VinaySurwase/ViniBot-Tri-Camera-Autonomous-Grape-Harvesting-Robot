#!/bin/bash
# =============================================================================
# setup_phoenix.sh — GrapeBot v2 Pi 5 Setup
# Run from: /home/phoenix/bot/
# =============================================================================

set -e

PROJECT="/home/phoenix/bot"
VENV="$PROJECT/venv"

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
_ok()   { echo -e "${GRN}  ✓  $1${NC}"; }
_warn() { echo -e "${YLW}  ⚠  $1${NC}"; }
_err()  { echo -e "${RED}  ✗  $1${NC}"; }
_step() { echo -e "\n${CYN}─── $1 ───${NC}"; }

echo -e "\n${CYN}══════════════════════════════════════════════${NC}"
echo -e "${CYN}  GrapeBot v2  |  phoenix  |  Pi 5 Setup${NC}"
echo -e "${CYN}══════════════════════════════════════════════${NC}"

# ── 1. System packages ────────────────────────────────────────────────────────
_step "System packages"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv python3-opencv \
    libopencv-dev libopenblas-dev \
    libcamera-dev libcamera-tools \
    python3-picamera2 \
    v4l-utils libv4l-dev \
    git build-essential cmake pkg-config \
    2>/dev/null
_ok "System packages installed"

# ── 2. Virtual environment ────────────────────────────────────────────────────
_step "Python venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV" --system-site-packages
    # --system-site-packages: inherits picamera2 which is system-installed
    _ok "venv created (with system-site-packages for picamera2)"
else
    _warn "venv exists — skipping"
fi

source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PROJECT/requirements.txt" --quiet
_ok "Python deps installed"

# ── 3. Camera check ───────────────────────────────────────────────────────────
_step "Camera detection"

# Pi cameras
CAM_COUNT=$(python3 -c "
from picamera2 import Picamera2
cams = Picamera2.global_camera_info()
print(len(cams))
" 2>/dev/null || echo "0")

if [ "$CAM_COUNT" -ge 2 ]; then
    _ok "Found $CAM_COUNT Pi Camera(s)"
else
    _warn "Found $CAM_COUNT Pi Camera — expected 2 (V3 modules)"
    _warn "Check ribbon cables and enable in raspi-config → Interface Options → Camera"
fi

# USB camera
USB_CAMS=$(v4l2-ctl --list-devices 2>/dev/null | grep -c "video" || echo "0")
if [ "$USB_CAMS" -gt 0 ]; then
    _ok "USB camera detected"
    v4l2-ctl --list-devices 2>/dev/null | head -6
else
    _warn "No USB camera detected — plug in USB cam and recheck"
fi

# ── 4. Serial permissions ─────────────────────────────────────────────────────
_step "Serial (STM32)"
if ! id -nG phoenix | grep -qw dialout; then
    sudo usermod -aG dialout phoenix
    _warn "Added phoenix to dialout — log out & in for this to take effect"
else
    _ok "phoenix already in dialout group"
fi

# ── 5. Directory structure ────────────────────────────────────────────────────
_step "Directories"
mkdir -p "$PROJECT/weights/det_ncnn_model"
mkdir -p "$PROJECT/weights/seg_ncnn_model"
mkdir -p "$PROJECT/calibration"
mkdir -p "$PROJECT/logs"
_ok "Directories ready"

# ── 6. Weights check ──────────────────────────────────────────────────────────
_step "NCNN Weights"
if ls "$PROJECT/weights/det_ncnn_model/"*.param 2>/dev/null | head -1 | grep -q ".param"; then
    _ok "Detection NCNN weights found"
else
    _warn "Detection weights missing → copy to weights/det_ncnn_model/"
fi

if ls "$PROJECT/weights/seg_ncnn_model/"*.param 2>/dev/null | head -1 | grep -q ".param"; then
    _ok "Segmentation NCNN weights found"
else
    _warn "Segmentation weights missing → copy to weights/seg_ncnn_model/"
fi

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
_step "Smoke test"
python3 - <<'PYEOF'
errors = []
for pkg, import_name in [
    ("picamera2", "picamera2"),
    ("cv2", "cv2"),
    ("numpy", "numpy"),
    ("ultralytics", "ultralytics"),
    ("yaml", "yaml"),
    ("websockets", "websockets"),
    ("serial", "serial"),
]:
    try:
        m = __import__(import_name)
        ver = getattr(m, "__version__", "ok")
        print(f"  ✓  {pkg:<15} {ver}")
    except ImportError as e:
        errors.append(f"{pkg}: {e}")

if errors:
    print("\n  ERRORS:")
    for e in errors:
        print(f"  ✗  {e}")
    import sys; sys.exit(1)
else:
    print("\n  All imports OK.")
PYEOF
_ok "Smoke test passed"

# ── 8. Systemd service (optional) ─────────────────────────────────────────────
_step "Systemd service (optional)"
SERVICE="/etc/systemd/system/grapebot.service"
if [ ! -f "$SERVICE" ]; then
    sudo tee "$SERVICE" > /dev/null <<EOF
[Unit]
Description=GrapeBot v2
After=network.target

[Service]
User=phoenix
WorkingDirectory=/home/phoenix/bot
ExecStart=/home/phoenix/bot/venv/bin/python main.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    _ok "Systemd service created: grapebot.service"
    _warn "To enable on boot: sudo systemctl enable grapebot"
    _warn "To start now:      sudo systemctl start grapebot"
    _warn "Logs:              journalctl -u grapebot -f"
else
    _warn "Service file already exists — skipping"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${CYN}══════════════════════════════════════════════${NC}"
echo -e "${GRN}  Setup complete for phoenix!${NC}"
echo -e "${CYN}══════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "    1. Copy NCNN weights into weights/det_ncnn_model/ and weights/seg_ncnn_model/"
echo "    2. Stereo calibration:"
echo "         source venv/bin/activate"
echo "         python calibration/run_calibration.py"
echo "    3. Tune HSV under vineyard lighting:"
echo "         python tools/tune_hsv.py --camera picamera2"
echo "    4. Test all subsystems:"
echo "         python tools/test_subsystem.py all --show"
echo "    5. Start the bot (dry-run first):"
echo "         python main.py --dry-run --debug"
echo "    6. Connect Android app to ws://$(hostname -I | awk '{print $1}'):8765"
echo ""