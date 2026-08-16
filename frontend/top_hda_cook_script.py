import os
import time

import pdg
import requests


def cook_trellis_item(work_item):
    """
    PDG Work Item Cook Cycle logic for the HouTrellis TOP HDA.
    """
    # 1. Fetch values from the Node's UI parameter interface
    node = work_item.holder.node

    api_url = node.evalParm("api_url")
    image_path = node.evalParm("image_path")

    # Resolve relative paths ($HIP, $JOB, etc.) to absolute paths
    image_path = (
        os.path.abspath(hou.expandString(image_path))
        if "hou" in globals()
        else os.path.abspath(image_path)
    )

    seed = int(node.evalParm("seed"))
    ss_steps = int(node.evalParm("ss_steps"))
    ss_strength = float(node.evalParm("ss_strength"))
    slat_steps = int(node.evalParm("slat_steps"))
    slat_strength = float(node.evalParm("slat_strength"))
    mesh_simplify = float(node.evalParm("mesh_simplify"))
    texture_size = int(node.evalParm("texture_size"))

    # Polling parameters
    poll_interval = 2.0
    timeout = 600.0  # 5 minutes max

    if not os.path.exists(image_path):
        print(f"Error: Input image file does not exist at: {image_path}")
        work_item.setFailed()
        return

    # 2. Package the JSON request payload
    payload = {
        "image_path": image_path,
        "seed": seed,
        "ss_sampling_steps": ss_steps,
        "ss_guidance_strength": ss_strength,
        "slat_sampling_steps": slat_steps,
        "slat_guidance_strength": slat_strength,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    }

    print(f"Triggering TRELLIS generation on: {api_url}")
    generate_endpoint = f"{api_url.rstrip('/')}/generate"

    try:
        # Trigger generation
        response = requests.post(generate_endpoint, json=payload, timeout=10)
        response.raise_for_status()

        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"Task queued successfully. Task ID: {task_id}")

        # 3. Poll status until completion
        status_endpoint = f"{api_url.rstrip('/')}/status/{task_id}"
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                print(f"Error: Generation timed out after {timeout} seconds.")
                work_item.setFailed()
                return

            status_response = requests.get(status_endpoint, timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data["status"]

            if status == "completed":
                glb_path = status_data["output_path"]
                print(f"Success! Model generated at: {glb_path}")

                # 4. Attach the generated GLB back to the PDG work item!
                # Downstream nodes (like USD / LOPs or SOPs) can now reference this path
                work_item.addResultData(glb_path, "file/glb", 0)
                break

            elif status == "failed":
                error_msg = status_data.get("error", "Unknown backend error")
                print(f"Error: TRELLIS Backend failed generation: {error_msg}")
                work_item.setFailed()
                return

            elif status in ["queued", "processing"]:
                print(f"State: {status}... Polling in {poll_interval}s")
                time.sleep(poll_interval)
            else:
                print(f"Error: Unexpected status value: {status}")
                work_item.setFailed()
                return

    except Exception as e:
        print(f"Exception raised during cook: {e}")
        work_item.setFailed()


# Execution entry point inside Houdini PDG
cook_trellis_item(work_item)
