# HouTrellis 🚀
An open-source integration bridging Microsoft's **TRELLIS** (3D Generative AI) directly into **SideFX Houdini** using an asynchronous out-of-process architecture.

Designed for high-performance VFX and games studios, this pipeline runs heavy machine learning workloads in an isolated external Python service (perfect for target RTX 4090/A100 server configurations), avoiding any C++/CUDA runtime DLL library conflicts inside Houdini's Python environment.

---

## 🏗️ Architecture Design

```mermaid
graph LR
    H_PDG[Houdini PDG Network] -->|1. Async HTTP POST /generate| API[FastAPI Local Server]
    API -->|2. Task ID / Queue| H_PDG
    API -->|3. Runs Inference| GPU[RTX 4090 / CUDA]
    H_PDG -->|4. Poll /status/{id}| API
    GPU -->|5. Exports GLB| GLB[outputs/.glb]
    API -->|6. Status: Completed| H_PDG
    H_PDG -->|7. Bind GLB to Work Item| SOP[Houdini SOP / Karma Setup]
```

1. **AI Backend (Python Service):** A lightweight FastAPI local microservice running on your AI hardware. It keeps the heavy model weights in GPU memory and exposes robust REST endpoints.
2. **Houdini PDG Frontend:** A Python Script TOP node handles parallel asynchronously dispatched requests and poll logic to keep the Houdini UI fluid.
3. **Karma/MaterialX SOP:** Standard Houdini nodes unpack the model, assign MaterialX surface shaders, and link textures automatically.

---

## 📦 1. Installation & Standalone Setup

### Prerequisites
- Python 3.9 or higher
- NVIDIA CUDA Toolkit & GPU (for real TRELLIS inference)

### Installation Steps

1. **Clone and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd houtrellis
   ```

2. **Initialize and install requirements:**
   ```bash
   python3 -m venv backend/venv
   source backend/venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Install TRELLIS & PyTorch Dependencies:**
   *(Run inside your active virtual environment)*
   ```bash
   # Install PyTorch with your CUDA flavor
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install TRELLIS dependencies (refer to official TRELLIS repo for exact model setup)
   pip install git+https://github.com/Microsoft/TRELLIS.git
   ```

---

## 🧪 2. Running Standalone Testing

We have built a fully simulated end-to-end integration test. You can test the backend pipeline without opening Houdini or having a GPU.

```bash
chmod +x test_pipeline.sh
./test_pipeline.sh
```

---

## 🎬 3. Inside Houdini (TOP / SOP Workflow)

### TOP Node Integration
1. Inside a **TOP Network**, place a **Python Script** or **Generic Generator** TOP node.
2. Add the following parameters to the node's user interface:
   - `api_url` (String) -> `http://127.0.0.1:8000`
   - `image_path` (File Path) -> path to your reference image.
   - `seed` (Integer), `ss_steps` (Integer), `ss_strength` (Float)
   - `slat_steps` (Integer), `slat_strength` (Float)
3. Copy the script content from `houdini/pdg_trellis_node.py` and paste it inside the **Script** tab of your TOP node.
4. Cook the TOP node. It will asynchronously generate the 3D assets on your RTX 4090 and append the output path back to PDG!

### SOP Import Setup
Refer to `houdini/sop_import_setup.md` for a complete visual network guide to procedurally import, unpack, and texture-link the generated GLB using native SOPs and **MaterialX** for Karma.
