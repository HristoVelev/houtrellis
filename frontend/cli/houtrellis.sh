#!/bin/bash
# =====================================================================
# HouTrellis: Standalone Multiplatform Linux/macOS Launcher
# =====================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$( dirname "$DIR" )" )"

# Ensure the launcher's backend is in sys.path
export PYTHONPATH="$ROOT_DIR/backend:$PYTHONPATH"

# Execute the modular CLI module using our virtual environment's Python interpreter
"$ROOT_DIR/backend/venv/bin/python" "$ROOT_DIR/backend/cli.py" "$@"
