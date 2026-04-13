#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Building frontend assets..."
build_frontend_here
echo "Starting unified AIOps Delight app..."
echo "App: http://127.0.0.1:5001"
run_backend_here
