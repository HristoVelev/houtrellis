import os
import time

import requests

# Inside a Houdini TOP node, we can access work_item attributes.
# This script is designed to run within a Python Script TOP or Generic Generator TOP.
# We'll use Houdini's `pg` (pdg) library to fetch attributes if running inside Houdini,
# but structure it cleanly so it's easy to copy-paste.


def generate_trellis_mesh(
    api_url: str,
    image_path: str,
    seed: int,
    ss_steps: int,
    ss_strength: float,
    slat_steps: int,
    slat_strength: float,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024,
    poll_interval: float = 1.0,
    timeout: float = 300.0,
) -> str:
    """
    Sends a generation request to the HouTrellis backend and polls until completion.
    Returns the absolute path to the generated GLB mesh.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Prepare payload
    payload = {
        "image_path": os.path.abspath(image_path),
        "seed": seed,
        "ss_sampling_steps": ss_steps,
        "ss_guidance_strength": ss_strength,
        "slat_sampling_steps": slat_steps,
        "slat_guidance_strength": slat_strength,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    }

    # 1. Trigger the generation
    generate_endpoint = f"{api_url.rstrip('/')}/generate"
    print(f"Sending request to: {generate_endpoint}")
    response = requests.post(generate_endpoint, json=payload)
    response.raise_for_status()

    task_data = response.json()
    task_id = task_data["task_id"]
    print(f"Task successfully queued. Task ID: {task_id}")

    # 2. Poll for the result
    status_endpoint = f"{api_url.rstrip('/')}/status/{task_id}"
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds.")

        status_response = requests.get(status_endpoint)
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data["status"]

        if status == "completed":
            output_path = status_data["output_path"]
            print(f"Task completed! Output file: {output_path}")
            return output_path
        elif status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            raise RuntimeError(f"TRELLIS generation failed: {error_msg}")
        elif status in ["queued", "processing"]:
            print(f"Task status: {status}... sleeping for {poll_interval}s")
            time.sleep(poll_interval)
        else:
            raise ValueError(f"Unexpected task status: {status}")


# --- PDG Integration Entrypoint ---
# When running inside a Houdini TOP node, the global variable `self` and `work_item` are available.
if "work_item" in globals():
    # Fetch parameters from the Houdini TOP node interface
    # (Assuming these exist as parameters/attributes on the TOP node or work_item)
    node = self.node

    api_url = node.evalParm("api_url")
    # Resolve the input image file path from the previous work item or parameter
    image_path = node.evalParm("image_path")

    seed = int(node.evalParm("seed"))
    ss_steps = int(node.evalParm("ss_steps"))
    ss_strength = float(node.evalParm("ss_strength"))
    slat_steps = int(node.evalParm("slat_steps"))
    slat_strength = float(node.evalParm("slat_strength"))

    # Fetch newly wired quality/resolution parameters with standard safe fallbacks
    mesh_simplify = (
        float(node.evalParm("mesh_simplify")) if node.parm("mesh_simplify") else 0.95
    )
    texture_size = (
        int(node.evalParm("texture_size")) if node.parm("texture_size") else 1024
    )

    try:
        # Run generation
        glb_path = generate_trellis_mesh(
            api_url=api_url,
            image_path=image_path,
            seed=seed,
            ss_steps=ss_steps,
            ss_strength=ss_strength,
            slat_steps=slat_steps,
            slat_strength=slat_strength,
            mesh_simplify=mesh_simplify,
            texture_size=texture_size,
        )

        # Add output GLB file back to the PDG work item so downstream SOPs can read it
        work_item.addResultData(glb_path, "file/glb", 0)
        print("Successfully added output file to PDG work item.")

    except Exception as e:
        # Let PDG know the work item failed
        print(f"Error in HouTrellis PDG Task: {e}")
        work_item.setFailed()
