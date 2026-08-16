#!/bin/bash
# Standalone pipeline tester for HouTrellis

# Ensure we exit on error
set -e

echo "=== 1. Starting HouTrellis FastAPI Backend ==="
./backend/venv/bin/python backend/app.py &
BACKEND_PID=$!

# Ensure the backend process gets killed on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

# Wait 12 seconds for uvicorn to bind to port 8000
echo "Waiting for backend to start up..."
sleep 12

echo "=== 2. Running Standalone Client Test ==="
./backend/venv/bin/python backend/test_client.py "$@"

echo "=== Test Complete ==="
