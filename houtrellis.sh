#!/bin/bash
# =====================================================================
# HouTrellis: Standalone Multiplatform Linux/macOS Launcher
# =====================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure the launcher's backend is in sys.path
export PYTHONPATH="$DIR/backend:$PYTHONPATH"

# Execute the modular CLI module using our virtual environment's Python interpreter
"$DIR/backend/venv/bin/python" "$DIR/backend/cli.py" "$@"
