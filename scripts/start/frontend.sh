#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Building frontend assets..."
build_frontend_here
echo "Frontend build complete. Flask will serve the app from frontend/dist."
