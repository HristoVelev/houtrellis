import asyncio
import json
import os
import struct
import sys
import time
import traceback
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# Force TRELLIS to use xformers attention backend
# and native spconv algorithm. This prevents compiling heavy third-party CUDA kernels.
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

print("=== HouTrellis Backend Startup: Loading AI Libraries ===")
# Add the cloned TRELLIS repository to sys.path dynamically
backend_dir = os.path.dirname(os.path.abspath(__file__))
trellis_repo_path = os.path.join(backend_dir, "TRELLIS")
if os.path.exists(trellis_repo_path) and trellis_repo_path not in sys.path:
    sys.path.append(trellis_repo_path)
    print(f"Added TRELLIS repo to sys.path: {trellis_repo_path}")

try:
    print("Attempting to import torch...")
    import torch

    print(
        f"PyTorch imported successfully (Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()})"
    )

    # Monkeypatch xformers to solve compatibility with BlockDiagonalMask
    try:
        import xformers.ops.fmha
        import xformers.ops.fmha.attn_bias

        if hasattr(xformers.ops.fmha.attn_bias, "BlockDiagonalMask") and not hasattr(
            xformers.ops.fmha, "BlockDiagonalMask"
        ):
            xformers.ops.fmha.BlockDiagonalMask = (
                xformers.ops.fmha.attn_bias.BlockDiagonalMask
            )
            print("Monkeypatched xformers.ops.fmha.BlockDiagonalMask successfully!")
    except Exception as e:
        print(f"No xformers monkeypatch needed or failed: {e}")

    print("Attempting to import PIL...")
    from PIL import Image

    print("Attempting to import TRELLIS components...")
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    HAS_TRELLIS = True
    print("SUCCESS: All TRELLIS AI modules loaded successfully!")
except ImportError as e:
    print(
        "\n⚠️ WARNING: Could not import TRELLIS dependencies. Running in SIMULATED FALLBACK mode."
    )
    print("Error details:")
    traceback.print_exc()
    print("========================================================================\n")
    torch = None
    HAS_TRELLIS = False

app = FastAPI(title="HouTrellis API")

# Global pipeline cache to avoid reloading weights on every request
GLOBAL_PIPELINE = None


def write_minimal_glb(output_path):
    """
    Generates a mathematically valid, minimal GLB file representing a 3D tetrahedron.
    This acts as a solid, compliant file for downstream pipeline tests (Houdini gltf import, etc.).
    """
    # 4 Vertices of a 3D Tetrahedron (VEC3)
    vertices = [
        0.0,
        0.5,
        0.0,  # Top Vertex
        -0.5,
        -0.5,
        -0.5,  # Base Bottom-Left-Back
        0.5,
        -0.5,
        -0.5,  # Base Bottom-Right-Back
        0.0,
        -0.5,
        0.5,  # Base Bottom-Front
    ]
    # 12 indices representing 4 triangular faces
    indices = [
        1,
        3,
        2,  # Base face
        0,
        1,
        2,  # Left-back face
        0,
        2,
        3,  # Right face
        0,
        3,
        1,  # Left-front face
    ]

    bin_data = bytearray()
    for v in vertices:
        bin_data.extend(struct.pack("<f", v))
    for i in indices:
        bin_data.extend(struct.pack("<H", i))

    gltf_json = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 48,
                "target": 34962,  # GL_ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": 48,
                "byteLength": 24,
                "target": 34963,  # GL_ELEMENT_ARRAY_BUFFER
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # GL_FLOAT
                "count": 4,
                "type": "VEC3",
                "max": [0.5, 0.5, 0.5],
                "min": [-0.5, -0.5, -0.5],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5123,  # GL_UNSIGNED_SHORT
                "count": 12,
                "type": "SCALAR",
                "max": [3],
                "min": [0],
            },
        ],
        "buffers": [{"byteLength": 72}],
    }

    json_str = json.dumps(gltf_json, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    align_len = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * align_len

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<4sII", b"glTF", 2, total_length)

    chunk0_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    chunk1_header = struct.pack("<I4s", len(bin_data), b"BIN\x00")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(chunk0_header)
        f.write(json_bytes)
        f.write(chunk1_header)
        f.write(bin_data)


def get_pipeline():
    global GLOBAL_PIPELINE
    if not HAS_TRELLIS:
        return None
    if GLOBAL_PIPELINE is None:
        print("Loading TRELLIS model weights onto GPU...")
        GLOBAL_PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        GLOBAL_PIPELINE.cuda()
        print("TRELLIS weights loaded successfully!")
    return GLOBAL_PIPELINE


# Storage for task status
tasks = {}


class GenerateRequest(BaseModel):
    task_id: Optional[str] = None  # Accept pre-generated Client-side Task IDs
    image_path: str
    seed: int = 42
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12
    mesh_simplify: float = 0.95
    texture_size: int = 1024


class TaskStatus(BaseModel):
    task_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None


class LogRedirector:
    """Context manager to redirect stdout/stderr to a task log file in real-time."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = None
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.log_file = open(
            self.log_path, "w", buffering=1
        )  # Buffering=1 for line-by-line write
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.log_file
        sys.stderr = self.log_file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        if self.log_file:
            self.log_file.close()


def run_trellis_inference(task_id: str, request: GenerateRequest):
    """Runs TRELLIS inference, redirecting logs to outputs/{task_id}.log"""
    log_file = os.path.abspath(f"backend/outputs/{task_id}.log")
    output_file = os.path.abspath(f"backend/outputs/{task_id}.glb")

    with LogRedirector(log_file):
        try:
            tasks[task_id]["status"] = "processing"

            if HAS_TRELLIS and torch is not None and torch.cuda.is_available():
                print(f"[{task_id}] Starting Real TRELLIS Generation...")
                pipeline = get_pipeline()

                image = Image.open(request.image_path)
                outputs = pipeline.run(
                    image,
                    seed=request.seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=True,
                    sparse_structure_sampler_params={
                        "steps": request.ss_sampling_steps,
                        "cfg_strength": request.ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": request.slat_sampling_steps,
                        "cfg_strength": request.slat_guidance_strength,
                    },
                )

                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=request.mesh_simplify,
                    texture_size=request.texture_size,
                    fill_holes=True,
                )
                glb.export(output_file)
                print(f"SUCCESS: Generated GLB mesh completely!")

            else:
                print(f"[{task_id}] Simulated generation starting...")
                time.sleep(5)
                write_minimal_glb(output_file)
                print(f"SUCCESS: Generated GLB mesh completely!")

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["output_path"] = output_file

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERROR: Generation failed:\n{error_details}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)


@app.post("/generate", response_model=TaskStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    # Use client-side generated Task ID if present, otherwise generate new
    task_id = request.task_id if request.task_id else str(uuid.uuid4())
    tasks[task_id] = {"status": "queued", "output_path": None, "error": None}

    # Execute in background, returning task_id instantly so Houdini can start log monitoring
    background_tasks.add_task(run_trellis_inference, task_id, request)

    return {
        "task_id": task_id,
        "status": "queued",
    }


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": tasks[task_id]["status"],
        "output_path": tasks[task_id]["output_path"],
        "error": tasks[task_id]["error"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
