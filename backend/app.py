import asyncio
import json
import os
import struct
import sys
import time
import traceback
import uuid
from typing import Optional

# Force TRELLIS to use xformers attention backend
# and native spconv algorithm. This prevents compiling heavy third-party CUDA kernels.
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

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

    # Pack binary arrays: 4 vertices * 3 floats * 4 bytes = 48 bytes
    # 12 indices * 2 bytes (unsigned short) = 24 bytes
    # Total BIN buffer size = 72 bytes
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
    # Pad JSON chunk with spaces to 4-byte alignment
    align_len = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * align_len

    # Binary GLB format headers:
    # 12-byte File Header: Magic (b'glTF'), Version (2), Total Length
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<4sII", b"glTF", 2, total_length)

    # 8-byte Chunk 0 (JSON) Header: Length, Type (b'JSON')
    chunk0_header = struct.pack("<I4s", len(json_bytes), b"JSON")

    # 8-byte Chunk 1 (BIN) Header: Length, Type (b'BIN\x00')
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
        # Load pre-trained TRELLIS pipeline
        GLOBAL_PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        GLOBAL_PIPELINE.cuda()
        print("TRELLIS weights loaded successfully!")
    return GLOBAL_PIPELINE


# Storage for task status
tasks = {}


class GenerateRequest(BaseModel):
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


def run_trellis_inference(task_id: str, request: GenerateRequest):
    """
    Runs TRELLIS inference or falls back to simulation mode.
    """
    try:
        tasks[task_id]["status"] = "processing"
        output_file = os.path.abspath(f"backend/outputs/{task_id}.glb")

        if HAS_TRELLIS and torch is not None and torch.cuda.is_available():
            print(f"[{task_id}] Running real TRELLIS inference...")
            pipeline = get_pipeline()

            # Load and preprocess input image
            image = Image.open(request.image_path)

            # Run the generative pipeline with parameters from the Houdini TOP node
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

            # Export output to standard GLB format
            # trellis.utils.postprocessing_utils.to_glb wraps mesh extraction and texture generation
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=request.mesh_simplify,  # Simplify mesh dynamically from user request
                texture_size=request.texture_size,  # Texture resolution dynamically from user request
                fill_holes=True,
            )
            glb.export(output_file)
            print(
                f"[{task_id}] Real TRELLIS generation completed! Saved to {output_file}"
            )

        else:
            print(
                f"[{task_id}] Real TRELLIS not available. Running simulated generation..."
            )
            # Simulated processing time
            time.sleep(5)

            # Generate a mathematically perfect, valid GLB file
            write_minimal_glb(output_file)
            print(
                f"[{task_id}] Simulated generation completed! Saved mathematically valid GLB: {output_file}"
            )

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["output_path"] = output_file

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"[{task_id}] Error during generation:\n{error_details}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


@app.post("/generate", response_model=TaskStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued", "output_path": None, "error": None}

    background_tasks.add_task(run_trellis_inference, task_id, request)

    return {"task_id": task_id, "status": "queued"}


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
