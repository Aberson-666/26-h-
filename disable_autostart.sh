#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="tspi-vision.service"

sudo systemctl stop "$SERVICE_NAME" || true
sudo systemctl disable "$SERVICE_NAME" || true
sudo rm -f "/etc/systemd/system/$SERVICE_NAME"
sudo systemctl daemon-reload

echo "disabled: $SERVICE_NAME"
