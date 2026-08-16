# HouTrellis Pipeline State 🧠
This document serves as the master context and pipeline blueprint for the **HouTrellis** open-source project. If the agent conversation context window is reset, read this file first to instantly re-initialize 100% of the project's background, resolved bugs, compilation tricks, and upcoming developmental steps.

---

## 🏗️ 1. Global Pipeline Architecture

The pipeline uses a **Strategy / Provider Design Pattern** with a fully decoupled out-of-process model-agnostic API. This isolates SideFX Houdini from heavy machine learning library installations and GPU CUDA collisions.

```mermaid
graph LR
    H_TOP[Houdini TOP Network] -->|1. Async POST /api/v1/txt2img| API[FastAPI Local Server]
    API -->|2. Task ID / Queue| H_TOP
    API -->|3. Starts Background Job| GPU[RTX 4090 / CUDA]
    H_TOP -->|4. Tails Local Log File| Log[outputs/*.log]
    GPU -->|5. Writes stdout/TeeStream| Log
    GPU -->|6. Exports GLB| GLB[outputs/*.glb]
    H_TOP -->|7. Polls /status/{id}| API
    H_TOP -->|8. Binds GLB to Task| SOP[Houdini Solaris / LOPs]
```

### 📁 Unified Multi-Platform Directory Structure
```text
houtrellis/
├── backend/                     # Modular local AI microservice
│   ├── app_core/                # Custom backend application
│   │   ├── api/ v1/ endpoints/  # API Routers (txt2img, img23d)
│   │   ├── core/                # vram.py (Active memory hot-swapping)
│   │   └── providers/           # Strategy Interfaces (Base, SDXL, Trellis, Hunyuan)
│   ├── third_party/             # Decoupled large cloned external repositories
│   │   ├── TRELLIS/             # Cloned Microsoft TRELLIS
│   │   └── mip-splatting/       # Cloned Mip-Splatting (for Trellis textures)
│   │   └── Hunyuan3D/           # Cloned Tencent Hunyuan3D-1
│   ├── outputs/                 # Generated PNG reference art, GLBs, and task log files
│   ├── app.py                   # FastAPI bootstrapper
│   └── test_runner.py           # Declarative YAML pipeline executor
├── frontend/                    # Consolidated client front-ends
│   ├── cli/                     # CLI suite
│   │   ├── houtrellis.bat       # Windows Launcher
│   │   ├── houtrellis.sh        # Linux/macOS Launcher
│   │   └── test_pipeline.yml    # Declarative YAML tests (chains SDXL -> Trellis-1 -> Trellis-2)
│   └── houdini/                 # Houdini Digital Assets
│       ├── build_hda.py         # HDA compiler automator (via hython)
│       ├── houtrellis_v05.hda   # Compiled self-healing HDA
│       ├── top_hda_cook_script.py # Internal node cook logic
│       └── sop_import_setup.md  # Solaris/MaterialX setup guides
├── setup.py                     # Root-level Package Installer
└── test_pipeline.sh             # Master startup and test trigger script (Probes port 8000)
```

---

## 🛠️ 2. Resolved Production Bugs & Compiler Tricks

During the "vibe coding" phase on our Rocky Linux 9 / CUDA 13.1 / RTX 4090 workstation, we successfully hunted down and smashed five critical bottlenecks:

### Bug A: Python Path Environment Variable Collisions
*   **Symptom:** Running the virtualenv Python from a Houdini Shell/Generic Generator TOP node crashed immediately with `Fatal Python error: ModuleNotFoundError: No module named 'encodings'`.
*   **Cause:** The spawned subprocess inherited Houdini's active `PYTHONHOME` and `PYTHONPATH` environment variables, hijacking the venv paths.
*   **Resolution:** Added `unset PYTHONHOME` and `unset PYTHONPATH` to the very top of our HDA's `start_server` Shell TOP command.

### Bug B: PyTorch CUDA Version Mismatches
*   **Symptom:** Compiling custom C++/CUDA extensions (`nvdiffrast`, `diff-gaussian-rasterization`, `pytorch3d`) threw a version mismatch error: `The detected CUDA version (13.1) mismatches the version that was used to compile PyTorch (12.4).`
*   **Resolution:** Modified PyTorch's native package check inside `/backend/venv/lib/python3.11/site-packages/torch/utils/cpp_extension.py`. We surgically monkeypatched `_check_cuda_version` to simply `return` immediately on cook, allowing the system compiler to build the kernels flawlessly on the RTX 4090!

### Bug C: xformers BlockDiagonalMask AttributeErrors
*   **Symptom:** Running real TRELLIS inference threw `AttributeError: module 'xformers.ops.fmha' has no attribute 'BlockDiagonalMask'`.
*   **Cause:** In modern `xformers>=0.0.23`, the `BlockDiagonalMask` class was relocated to the `attn_bias` submodule.
*   **Resolution:** Added a clean startup monkeypatch in `backend/app_core/providers/trellis_provider.py` (and `trellis2_provider.py`) before TRELLIS imports are run:
    ```python
    import xformers.ops.fmha
    import xformers.ops.fmha.attn_bias
    xformers.ops.fmha.BlockDiagonalMask = xformers.ops.fmha.attn_bias.BlockDiagonalMask
    ```

### Bug D: Flash_attn Import Errors & Environment Race
*   **Symptom:** Generating 3D meshes crashed with `ModuleNotFoundError: No module named 'flash_attn'`.
*   **Cause:** Setting `os.environ["ATTN_BACKEND"] = "xformers"` inside the `load_model()` method was executed *after* the `trellis.pipelines` modules had already been imported on server startup, causing the system to fallback to `flash_attn`.
*   **Resolution:** Set `os.environ["ATTN_BACKEND"] = "xformers"` at the **absolute top** of our `trellis_provider.py` module, well before any package imports can execute.

### Bug E: Relative Subcheckpoint 404 Errors (Trellis-1 vs Trellis-2)
*   **Symptom:** Loading TRELLIS-2 (4B) sub-checkpoints threw a 404 error because the path split on `'/'` default mapped back to TRELLIS-1 S3 directories.
*   **Resolution:** Modified `backend/third_party/TRELLIS/trellis/models/__init__.py` to dynamically fallback to `microsoft/TRELLIS-image-large` (Trellis-1) if the submodel path segments are `< 3`, while preserving the dynamic parent directory assembly for Trellis-2 (`microsoft/TRELLIS.2-4B`).

---

## 🔄 3. Memory & VRAM Management Strategy

To run multiple generative pipelines on a single RTX 4090 (24GB VRAM) without risk of GPU Out-Of-Memory (OOM) errors, the backend implements a strict, sequential **weight-offloading and VRAM flushing strategy** (`backend/app_core/core/vram.py`):

1.  **Before Loading a 2D Model (SDXL):** The server calls `vram.unload_3d()`. This removes any active 3D models, triggers Python's garbage collector (`gc.collect()`), and clears PyTorch's CUDA cache (`torch.cuda.empty_cache()`), freeing up full VRAM before booting SDXL.
2.  **Before Loading a 3D Model (Trellis / Hunyuan):** The server calls `vram.unload_2d()`. This offloads the SDXL base models from GPU memory, flushes PyTorch CUDA cache, and frees up full VRAM before booting the 3D pipelines.

---

## 🚀 4. How to Run the Environment

### A. Starting the AI Microservice Backend
```bash
# Activate the Python 3.11 environment
cd /home/admin/houtrellis
source backend/venv/bin/activate

# Launch the FastAPI server
python backend/app.py
```

### B. Executing the Declarative CLI YAML Suite (SDXL -> Trellis-1 -> Trellis-2)
```bash
# On Linux:
./frontend/cli/houtrellis.sh run frontend/cli/test_pipeline.yml

# On Windows:
frontend\cli\houtrellis.bat run frontend\cli\test_pipeline.yml
```

### C. Running Tencent Hunyuan3D Pipeline Tests
```bash
# Run the Hunyuan3D test script (Bootstraps server, waits, and executes)
./test_hunyuan.sh
```

---

## 🔮 5. Next Steps & Roadmaps

1.  **Compile PyTorch3D for Hunyuan3D:**
    To run the real GPU Hunyuan3D-1 pipeline, the C++/CUDA compiler needs to finish building the PyTorch3D rasterizer and marching cubes kernels. Run this command inside your terminal virtualenv and let it cook to completion:
    ```bash
    CUB_HOME="/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cub" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
    ```
2.  **Patch Trellis-2 (4B) Class Constructors:**
    As soon as Microsoft merges their updated TRELLIS-2 (4B) codebase into the `microsoft/TRELLIS` repository, pull the updates inside `backend/third_party/TRELLIS` and resolve any minor keyword constructor mismatches (like `patch_size` defaulting to `1` and `initialization` configurations).
