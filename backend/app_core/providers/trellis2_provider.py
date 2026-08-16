import json
import os
import struct
import sys
import time
import traceback
from typing import Any, Dict

from ..core import vram
from .base import Base3DProvider

# Force TRELLIS-2 environment variables at the absolute top of the module
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

# Dynamically add the cloned TRELLIS repository to sys.path
provider_dir = os.path.dirname(os.path.abspath(__file__))
trellis_path = os.path.abspath(
    os.path.join(provider_dir, "..", "..", "third_party", "TRELLIS")
)
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.append(trellis_path)
    print(f"[TRELLIS-2 Provider] Appended TRELLIS repo to sys.path: {trellis_path}")

try:
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    HAS_TRELLIS = True
except ImportError:
    torch = None
    HAS_TRELLIS = False


class Trellis2Provider(Base3DProvider):
    """Microsoft TRELLIS-2 (Next-Gen) 3D Generative Mesh Provider."""

    def load_model(self) -> Any:
        # Check if already loaded in our global VRAM cache
        active_pipeline = vram.get_active_3d()
        if active_pipeline is not None and getattr(self, "_is_t2_loaded", False):
            return active_pipeline

        # Ensure we unload any competing models (including Trellis-1) before loading Trellis-2
        vram.unload_2d()
        vram.unload_3d()

        if HAS_TRELLIS and torch is not None and torch.cuda.is_available():
            print(
                "[TRELLIS-2] Loading TRELLIS-2 (microsoft/TRELLIS.2-4B) weights onto GPU..."
            )

            # Apply xformers fmha BlockDiagonalMask monkeypatch if needed
            try:
                import xformers.ops.fmha
                import xformers.ops.fmha.attn_bias

                if hasattr(
                    xformers.ops.fmha.attn_bias, "BlockDiagonalMask"
                ) and not hasattr(xformers.ops.fmha, "BlockDiagonalMask"):
                    xformers.ops.fmha.BlockDiagonalMask = (
                        xformers.ops.fmha.attn_bias.BlockDiagonalMask
                    )
            except Exception:
                pass

            # Load the pre-trained TRELLIS-2 next-gen pipeline
            pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            pipeline.cuda()

            # Cache the active pipeline and set flag indicating it's TRELLIS-2
            vram.set_active_3d(pipeline)
            self._is_t2_loaded = True

            print("[TRELLIS-2] TRELLIS-2 weights loaded successfully!")
            return pipeline
        return None

    def _write_minimal_glb(self, output_path: str):
        """Generates a mathematically valid, minimal GLB file representing a 3D tetrahedron."""
        vertices = [0.0, 0.5, 0.0, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.0, -0.5, 0.5]
        indices = [1, 3, 2, 0, 1, 2, 0, 2, 3, 0, 3, 1]
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
                {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
                {"buffer": 0, "byteOffset": 48, "byteLength": 24, "target": 34963},
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": 4,
                    "type": "VEC3",
                    "max": [0.5, 0.5, 0.5],
                    "min": [-0.5, -0.5, -0.5],
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5123,
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

    def generate_3d(
        self, image_path: str, seed: int, params: Dict[str, Any], output_path: str
    ) -> str:
        # Check if we should load the real pipeline or run in mock mode
        pipeline = None
        try:
            pipeline = self.load_model()
        except Exception as le:
            print(
                f"[TRELLIS-2] Could not load real TRELLIS-2 weights (might not be released/accessible yet): {le}"
            )
            pipeline = None

        if pipeline is not None and torch is not None:
            print(f"[TRELLIS-2] Running real TRELLIS-2 inference on: {image_path}...")

            # Load and preprocess input image
            image = Image.open(image_path)

            # Extract additional custom parameters (Trellis-2 optimized defaults)
            ss_steps = params.get("ss_sampling_steps", 12)
            ss_strength = params.get("ss_guidance_strength", 7.5)
            slat_steps = params.get("slat_sampling_steps", 12)
            slat_strength = params.get("slat_guidance_strength", 3.0)
            simplify = params.get("mesh_simplify", 0.95)
            texture_size = params.get("texture_size", 1024)

            # Run next-gen pipeline
            outputs = pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=True,
                sparse_structure_sampler_params={
                    "steps": ss_steps,
                    "cfg_strength": ss_strength,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_strength,
                },
            )

            # Export output to standard GLB format
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
                fill_holes=True,
            )
            glb.export(output_path)
            print(
                f"[TRELLIS-2] Next-Gen Mesh successfully generated and saved to: {output_path}"
            )

        else:
            print(
                "[TRELLIS-2] TRELLIS-2/CUDA not available. Simulating TRELLIS-2 generation fallback..."
            )
            time.sleep(5)
            self._write_minimal_glb(output_path)
            print(f"[TRELLIS-2] Simulated TRELLIS-2 GLB mesh saved to: {output_path}")

        return output_path
