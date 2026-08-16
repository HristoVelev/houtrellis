import gc

import torch

# Global reference caches for active pipelines
_ACTIVE_2D_PIPELINE = None
_ACTIVE_3D_PIPELINE = None


def get_active_2d():
    global _ACTIVE_2D_PIPELINE
    return _ACTIVE_2D_PIPELINE


def set_active_2d(pipeline):
    global _ACTIVE_2D_PIPELINE
    _ACTIVE_2D_PIPELINE = pipeline


def get_active_3d():
    global _ACTIVE_3D_PIPELINE
    return _ACTIVE_3D_PIPELINE


def set_active_3d(pipeline):
    global _ACTIVE_3D_PIPELINE
    _ACTIVE_3D_PIPELINE = pipeline


def flush_vram():
    """Forces Python garbage collection and clears PyTorch's CUDA cache."""
    print("Flushing CUDA VRAM...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("VRAM flush completed.")


def unload_2d():
    """Unloads the active 2D pipeline from GPU memory."""
    global _ACTIVE_2D_PIPELINE
    if _ACTIVE_2D_PIPELINE is not None:
        print("Unloading 2D model weights from GPU VRAM...")
        # Move to CPU first to break CUDA links, then delete reference
        try:
            _ACTIVE_2D_PIPELINE.to("cpu")
        except Exception:
            pass
        _ACTIVE_2D_PIPELINE = None
        flush_vram()


def unload_3d():
    """Unloads the active 3D pipeline from GPU memory."""
    global _ACTIVE_3D_PIPELINE
    if _ACTIVE_3D_PIPELINE is not None:
        print("Unloading 3D model weights from GPU VRAM...")
        # In Trellis, we can unload the model or empty references
        _ACTIVE_3D_PIPELINE = None
        flush_vram()
