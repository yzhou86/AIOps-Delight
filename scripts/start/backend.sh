#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Building frontend assets for the unified Flask app..."
build_frontend_here
echo "Starting unified AIOps Delight app on http://127.0.0.1:5005 ..."
run_backend_here
