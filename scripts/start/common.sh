#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

ensure_command() {
  local cmd="$1"
  local message="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$message" >&2
    exit 1
  fi
}

backend_cmd() {
  local python_cmd
  python_cmd="$(pick_python)"
  if [[ -z "$python_cmd" ]]; then
    echo "Python was not found on PATH. Install Python 3 first." >&2
    exit 1
  fi
  echo "$python_cmd app.py"
}

run_backend_here() {
  local python_cmd
  python_cmd="$(pick_python)"
  cd "$BACKEND_DIR"
  exec "$python_cmd" app.py
}

build_frontend_here() {
  ensure_command npm "npm was not found on PATH. Install Node.js and npm first."
  cd "$FRONTEND_DIR"
  npm run build
}
