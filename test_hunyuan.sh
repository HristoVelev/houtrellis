#!/bin/bash
# Standalone declarative YAML pipeline tester for HouTrellis (Hunyuan3D)

# Ensure we exit on error
set -e

# Clean up any previously orphaned backend servers
pkill -f "app.py" || true
pkill -f "uvicorn" || true

echo "=== 1. Starting HouTrellis FastAPI Backend ==="
./backend/venv/bin/python backend/app.py &
BACKEND_PID=$!

# Ensure the backend process gets killed on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT

# Wait for uvicorn to bind to port 8000 using a robust curl probing loop
echo "Waiting for backend server to bind to port 8000..."
for i in {1..20}; do
    if curl -s http://127.0.0.1:8000/ > /dev/null; then
        echo "=== Server is active and responding! ==="
        break
    fi
    sleep 1
done

echo "=== 2. Running Declarative YAML Pipeline Test (Hunyuan3D) ==="
./backend/venv/bin/python backend/test_runner.py frontend/cli/hunyuan.yml

echo "=== Test Complete ==="
