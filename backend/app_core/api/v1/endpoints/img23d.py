import os
import sys
import time
import traceback
import uuid
from typing import Optional

from app_core.core import vram
from app_core.providers.hunyuan_provider import HunyuanProvider
from app_core.providers.trellis_provider import TrellisProvider
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Global task storage
tasks_3d = {}


class Img23DRequest(BaseModel):
    task_id: Optional[str] = None
    provider: str = "trellis"  # "trellis", "hunyuan", etc.
    image_path: str
    seed: int = 42
    ss_sampling_steps: int = 12
    ss_guidance_strength: float = 7.5
    slat_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    mesh_simplify: float = 0.95
    texture_size: int = 1024


class Img23DResponse(BaseModel):
    task_id: str
    status: str
    mesh_path: Optional[str] = None
    log_path: Optional[str] = None
    error: Optional[str] = None


class LogRedirector:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = None
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.log_file = open(self.log_path, "w", buffering=1)
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.TeeStream(self.old_stdout, self.log_file)
        sys.stderr = self.TeeStream(self.old_stderr, self.log_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        if self.log_file:
            self.log_file.close()

    class TeeStream:
        def __init__(self, terminal, file):
            self.terminal = terminal
            self.file = file

        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.terminal.flush()
            self.file.flush()

        def flush(self):
            self.terminal.flush()
            self.file.flush()


def run_img23d_inference(task_id: str, request: Img23DRequest):
    log_file = os.path.abspath(f"backend/outputs/{task_id}.log")
    output_file = os.path.abspath(f"backend/outputs/{task_id}.glb")

    with LogRedirector(log_file):
        try:
            tasks_3d[task_id]["status"] = "processing"

            # Resolve Strategy Provider
            if request.provider.lower() == "trellis":
                provider = TrellisProvider()
            elif request.provider.lower() == "hunyuan":
                provider = HunyuanProvider()
            else:
                raise ValueError(f"Unknown 3D Provider: {request.provider}")

            # Generate 3D asset
            params = {
                "ss_sampling_steps": request.ss_sampling_steps,
                "ss_guidance_strength": request.ss_guidance_strength,
                "slat_sampling_steps": request.slat_sampling_steps,
                "slat_guidance_strength": request.slat_guidance_strength,
                "mesh_simplify": request.mesh_simplify,
                "texture_size": request.texture_size,
            }
            provider.generate_3d(request.image_path, request.seed, params, output_file)

            tasks_3d[task_id]["status"] = "completed"
            tasks_3d[task_id]["mesh_path"] = output_file

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERROR: 3D Generation failed:\n{error_details}")
            tasks_3d[task_id]["status"] = "failed"
            tasks_3d[task_id]["error"] = str(e)


@router.post("/img23d", response_model=Img23DResponse)
async def img23d(request: Img23DRequest, background_tasks: BackgroundTasks):
    task_id = request.task_id if request.task_id else str(uuid.uuid4())

    output_file = os.path.abspath(f"backend/outputs/{task_id}.glb")
    log_file = os.path.abspath(f"backend/outputs/{task_id}.log")

    tasks_3d[task_id] = {
        "status": "queued",
        "mesh_path": output_file,
        "log_path": log_file,
        "error": None,
    }

    # Execute in background thread
    background_tasks.add_task(run_img23d_inference, task_id, request)

    return {
        "task_id": task_id,
        "status": "queued",
        "mesh_path": output_file,
        "log_path": log_file,
        "error": None,
    }


@router.get("/img23d/status/{task_id}", response_model=Img23DResponse)
async def get_img23d_status(task_id: str):
    if task_id not in tasks_3d:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": tasks_3d[task_id]["status"],
        "mesh_path": tasks_3d[task_id].get("mesh_path"),
        "log_path": tasks_3d[task_id].get("log_path"),
        "error": tasks_3d[task_id].get("error"),
    }
