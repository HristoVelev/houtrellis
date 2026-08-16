import os
import sys

# Bootstrap setup: Ensure that the 'backend' parent directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

if __name__ == "__main__":
    import uvicorn

    # Import our brand new modular API router application directly
    from app_core.main import app

    uvicorn.run(app, host="0.0.0.0", port=8000)
