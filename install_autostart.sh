#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="tspi-vision.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="$(id -un)"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
TEMPLATE="$SCRIPT_DIR/tspi_autostart.service"
TARGET="/etc/systemd/system/$SERVICE_NAME"

if [ ! -f "$SCRIPT_DIR/detect_new01.py" ]; then
  echo "error: detect_new01.py not found in $SCRIPT_DIR"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found"
  exit 1
fi

tmp_file="$(mktemp)"
sed \
  -e "s|__WORKDIR__|$SCRIPT_DIR|g" \
  -e "s|__USER__|$USER_NAME|g" \
  -e "s|__HOME__|$USER_HOME|g" \
  "$TEMPLATE" > "$tmp_file"

sudo cp "$tmp_file" "$TARGET"
rm -f "$tmp_file"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "installed: $SERVICE_NAME"
echo "status: sudo systemctl status $SERVICE_NAME"
echo "logs: journalctl -u $SERVICE_NAME -f"
