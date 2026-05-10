#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ensure_command osascript "osascript was not found. This launcher is for macOS."
echo "Building frontend assets..."
build_frontend_here

BACKEND_CMD="$(backend_cmd)"

echo "Starting unified AIOps Delight app on macOS..."
osascript <<EOF
tell application "Terminal"
  activate
  do script "cd \"$BACKEND_DIR\"; $BACKEND_CMD"
end tell
EOF

echo "App: http://127.0.0.1:5005"
