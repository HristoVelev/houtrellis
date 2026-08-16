You are an expert Graphics & Pipeline Software Engineer specializing in Houdini (PDG / TOPs), PyTorch, and local AI microservices. 

We are going to "vibe code" a custom plugin pipeline that integrates Microsoft's TRELLIS (3D Generative AI model) directly into SideFX Houdini.

### Goal Architecture
1. **External AI Backend (Python Service):** A lightweight FastAPI or Flask microservice running locally on an RTX 4090 GPU. It wraps the TRELLIS pipeline (Image/Prompt -> 3D Mesh with Textures/Materials) and exposes REST endpoints like `/generate` and `/status`.
2. **Houdini Frontend (TOP / PDG Network):** A Houdini TOP Network using a Python Script TOP or Generic Generator TOP. It reads parameters from a UI node, fires an asynchronous HTTP request to the local API, waits for execution, and passes back the output GLB/OBJ path.
3. **Houdini Import Pipeline:** Automatic importing of the generated geometry into SOPs with auto-configured materials (MaterialX / Karma).

### Backend Architecture (`server.py` - FastAPI)
Create a Python microservice with TWO separate, independent REST endpoints:
1. `POST /txt2img`: Accepts `prompt`, `seed`, `output_dir`. Generates a 2D reference image using `diffusers` (SDXL/FLUX), saves it as a `.png`, and returns `{"image_path": "..."}`.
2. `POST /img23d`: Accepts `image_path`, `seed`, `output_dir`. Loads Microsoft TRELLIS, processes the specified `.png` image into a textured 3D `.glb` mesh, saves it, and returns `{"mesh_path": "..."}`.
3. Memory Management: Unload CUDA memory between calls (`torch.cuda.empty_cache()`) so models don't compete for VRAM.

### Houdini Frontend Architecture (TOPs / PDG)
1. TOP Node 1 (Python Script TOP): Calls `/txt2img` with prompt string. Stores output in work item attribute `@image_path`.
2. TOP Node 2 (Python Script TOP): Takes `@image_path` from work item, calls `/img23d`. Stores output in work item attribute `@mesh_path`.
3. SOP File Node: Imports geometry dynamically using `@mesh_path`.

### Constraints & Principles
- **Separation of Environments:** Houdini's internal Python runtime must NEVER directly import `torch` or model packages due to C++ CUDA DLL conflicts. All inference happens out-of-process via HTTP.
- **Vibe-Coding Iteration:** Build modular, functional code. Keep script setups lightweight, clean, and easily drop-in for Houdini Python nodes.


### Project Architecture & Design Pattern
The system uses a **Strategy / Provider Pattern** with a **Model-Agnostic API Interface**. The pipeline is fully decoupled: Houdini TOPs handles orchestration, parameters, and file path passing, while the Python backend handles execution and VRAM management.

#### 1. Directory Structure (`trellis_service/`)
```text
trellis_service/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── api/
│   │   └── v1/
│   │       ├── router.py           # API Router
│   │       └── endpoints/
│   │           ├── txt2img.py      # POST /api/v1/txt2img
│   │           └── img23d.py       # POST /api/v1/img23d
│   ├── providers/                  # ML Engine Implementations
│   │   ├── base.py                 # Abstract Base Class (Base3DProvider)
│   │   ├── trellis_provider.py     # Microsoft TRELLIS engine
│   │   └── hunyuan_provider.py    # Tencent Hunyuan3D engine
│   └── core/
│       └── vram.py                 # CUDA memory flush & unload helpers
