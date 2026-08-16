import os
import sys
import time
import traceback
import uuid
from typing import Optional

from app_core.core import vram
from app_core.providers.sdxl_provider import SDXLProvider
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Global task storage
tasks_2d = {}


class Txt2ImgRequest(BaseModel):
    task_id: Optional[str] = None
    provider: str = "sdxl"  # "sdxl", etc.
    prompt: str
    seed: int = 42
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25


class Txt2ImgResponse(BaseModel):
    task_id: str
    status: str
    image_path: Optional[str] = None
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


def run_txt2img_inference(task_id: str, request: Txt2ImgRequest):
    log_file = os.path.abspath(f"backend/outputs/{task_id}.log")
    output_file = os.path.abspath(f"backend/outputs/{task_id}.png")

    with LogRedirector(log_file):
        try:
            tasks_2d[task_id]["status"] = "processing"

            # Resolve Strategy Provider
            if request.provider.lower() == "sdxl":
                provider = SDXLProvider()
            else:
                raise ValueError(f"Unknown 2D Provider: {request.provider}")

            # Generate reference image
            params = {
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
            }
            provider.generate_2d(request.prompt, request.seed, params, output_file)

            tasks_2d[task_id]["status"] = "completed"
            tasks_2d[task_id]["image_path"] = output_file

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERROR: 2D Generation failed:\n{error_details}")
            tasks_2d[task_id]["status"] = "failed"
            tasks_2d[task_id]["error"] = str(e)


@router.post("/txt2img", response_model=Txt2ImgResponse)
async def txt2img(request: Txt2ImgRequest, background_tasks: BackgroundTasks):
    task_id = request.task_id if request.task_id else str(uuid.uuid4())

    output_file = os.path.abspath(f"backend/outputs/{task_id}.png")
    log_file = os.path.abspath(f"backend/outputs/{task_id}.log")

    tasks_2d[task_id] = {
        "status": "queued",
        "image_path": output_file,
        "log_path": log_file,
        "error": None,
    }

    # Execute in background thread
    background_tasks.add_task(run_txt2img_inference, task_id, request)

    return {
        "task_id": task_id,
        "status": "queued",
        "image_path": output_file,
        "log_path": log_file,
        "error": None,
    }


@router.get("/txt2img/status/{task_id}", response_model=Txt2ImgResponse)
async def get_txt2img_status(task_id: str):
    if task_id not in tasks_2d:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": tasks_2d[task_id]["status"],
        "image_path": tasks_2d[task_id].get("image_path"),
        "log_path": tasks_2d[task_id].get("log_path"),
        "error": tasks_2d[task_id].get("error"),
    }
