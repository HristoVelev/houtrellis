import os
import time
from typing import Any, Dict

from ..core import vram
from .base import Base2DProvider

try:
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    HAS_DIFFUSERS = True
except ImportError:
    torch = None
    HAS_DIFFUSERS = False


class SDXLProvider(Base2DProvider):
    """Stable Diffusion XL Concept Art Generator Provider."""

    def load_model(self) -> Any:
        # Check if already loaded in our global VRAM cache
        active_pipeline = vram.get_active_2d()
        if active_pipeline is not None:
            return active_pipeline

        # Ensure we unload any competing 3D models before loading SDXL
        vram.unload_3d()

        if HAS_DIFFUSERS and torch is not None and torch.cuda.is_available():
            print("[SDXL] Loading SDXL model weights onto GPU...")
            # We load SDXL-turbo or standard fast SDXL for lightweight VRAM footprints
            # stable-diffusion-xl-base-1.0 is default, but you can configure any model
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            pipeline.to("cuda")
            vram.set_active_2d(pipeline)
            print("[SDXL] SDXL loaded successfully!")
            return pipeline
        return None

    def generate_2d(
        self, prompt: str, seed: int, params: Dict[str, Any], output_path: str
    ) -> str:
        pipeline = self.load_model()

        if pipeline is not None and torch is not None:
            print(f"[SDXL] Generating 2D image for prompt: '{prompt}'...")

            # Extract additional custom parameters with standard defaults
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            steps = params.get("num_inference_steps", 25)

            generator = torch.Generator("cuda").manual_seed(seed)

            # Generate the image
            image = pipeline(
                prompt=prompt,
                generator=generator,
                width=width,
                height=height,
                num_inference_steps=steps,
            ).images[0]

            # Save the PNG
            image.save(output_path)
            print(f"[SDXL] Reference image saved completely to: {output_path}")

        else:
            print(
                "[SDXL] Diffusers/CUDA not available. Simulating 2D generation fallback..."
            )
            time.sleep(3)

            # Create a mock 100-byte transparent PNG file for Houdini testing
            with open(output_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 92)
            print(f"[SDXL] Simulated 2D image saved completely to: {output_path}")

        return output_path
