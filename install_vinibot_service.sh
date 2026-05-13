#!/bin/bash
# =============================================================================
# install_vinibot_service.sh
# Sets up ViniBot to run automatically in the background on startup, waiting
# for the Pi to connect to your phone's mobile hotspot.
# =============================================================================

set -e

SERVICE_PATH="/etc/systemd/system/vinibot.service"

echo "Creating ViniBot background service at $SERVICE_PATH..."

# Create the systemd service file
# 'network-online.target' ensures it waits for the Pi to successfully connect
# to your mobile hotspot before launching main.py.
sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=ViniBot Robot Service
Wants=network-online.target
After=network-online.target

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

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling vinibot service so it starts on every boot..."
sudo systemctl enable vinibot

echo "Starting vinibot service now..."
sudo systemctl restart vinibot

echo ""
echo "====================================================="
echo "  ✓ ViniBot service is now installed and running!"
echo "    It will automatically run main.py whenever"
echo "    your Pi boots and connects to your phone."
echo "====================================================="
echo "Useful Commands:"
echo "  View live logs:   journalctl -u vinibot -f"
echo "  Stop the bot:     sudo systemctl stop vinibot"
echo "  Restart the bot:  sudo systemctl restart vinibot"
echo ""
