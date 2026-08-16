import os
import sys

# Define HDA version suffix here
VERSION = "v01"


def build_trellis_top_hda():
    print(f"=== Creating HouTrellis TOP HDA Programmatically ({VERSION}) ===")

    # Ensure hou module is loaded (run inside hython)
    try:
        import hou
    except ImportError:
        print(
            "Error: This script must be run using Houdini's python interpreter (hython)!"
        )
        sys.exit(1)

    hda_dir = os.path.dirname(os.path.abspath(__file__))
    hda_path = os.path.join(hda_dir, f"houtrellis_{VERSION}.hda")

    # 1. Create a temporary network to scaffold our node
    topnet = hou.node("/out").createNode("topnet", "temp_topnet")

    # Create a TOP Subnet Node which will be the basis of our HDA
    hda_node = topnet.createNode("subnet", "houtrellis")

    # Create the internal Python Script TOP inside the subnet to do the work
    core_node = hda_node.createNode("pythonscript", "trellis_core")

    # Wire the internal nodes so they are connected between Subnet Input and Subnet Output
    subnet_input = hda_node.node("subnetinput1")
    subnet_output = hda_node.node("subnetoutput1")

    if subnet_input and subnet_output:
        print("Wiring internal Python Script TOP node between input and output...")
        core_node.setInput(0, subnet_input)
        subnet_output.setInput(0, core_node)

        # Position them nicely inside the HDA graph viewport
        subnet_input.setPosition(hou.Vector2(0, 2))
        core_node.setPosition(hou.Vector2(0, 0))
        subnet_output.setPosition(hou.Vector2(0, -2))

    # 2. Configure parameters for our HDA (on the parent subnet container node)
    group = hda_node.parmTemplateGroup()

    # Folder Structure (Tabs)
    server_folder = hou.FolderParmTemplate("server_tab", "Server")
    server_folder.addParmTemplate(
        hou.StringParmTemplate(
            "api_url", "API URL", 1, default_value=["http://127.0.0.1:8000"]
        )
    )

    input_folder = hou.FolderParmTemplate("input_tab", "Inputs")
    input_folder.addParmTemplate(
        hou.StringParmTemplate(
            "image_path", "Input Image", 1, string_type=hou.stringParmType.FileReference
        )
    )

    sampler_folder = hou.FolderParmTemplate("sampler_tab", "Samplers")
    sampler_folder.addParmTemplate(
        hou.IntParmTemplate("seed", "Seed", 1, default_value=[42])
    )
    sampler_folder.addParmTemplate(
        hou.IntParmTemplate("ss_steps", "Sparse Steps", 1, default_value=[12])
    )
    sampler_folder.addParmTemplate(
        hou.FloatParmTemplate("ss_strength", "Sparse Guidance", 1, default_value=[7.5])
    )
    sampler_folder.addParmTemplate(
        hou.IntParmTemplate("slat_steps", "Slat Steps", 1, default_value=[12])
    )
    sampler_folder.addParmTemplate(
        hou.FloatParmTemplate("slat_strength", "Slat Guidance", 1, default_value=[3.0])
    )

    quality_folder = hou.FolderParmTemplate("quality_tab", "Quality")
    quality_folder.addParmTemplate(
        hou.FloatParmTemplate("mesh_simplify", "Mesh Simplify", 1, default_value=[0.95])
    )
    quality_folder.addParmTemplate(
        hou.IntParmTemplate("texture_size", "Texture Size", 1, default_value=[1024])
    )

    # Add Folders to Group
    group.append(server_folder)
    group.append(input_folder)
    group.append(sampler_folder)
    group.append(quality_folder)

    hda_node.setParmTemplateGroup(group)

    # 3. Read our custom cook script and embed it inside the internal python script parameter
    cook_script_path = os.path.join(hda_dir, "top_hda_cook_script.py")
    with open(cook_script_path, "r") as f:
        cook_script_content = f.read()

    # We alter the internal cook script slightly to dynamically fetch from parent (HDA subnet)
    # if running inside a container, or fallback to local node.
    enhanced_cook_script = """# Dynamic Parameter Node Resolver
import os
import time
import requests
import pdg

def cook_trellis_item(work_item):
    node = work_item.holder.node
    # If we are inside an HDA Subnet container, parameters are on the parent node
    param_node = node.parent() if (node.parent() and node.parent().type().name() != "topnet") else node

    api_url = param_node.evalParm('api_url')
    image_path = param_node.evalParm('image_path')

    # Resolve relative paths ($HIP, $JOB, etc.) to absolute paths
    image_path = os.path.abspath(hou.expandString(image_path)) if 'hou' in globals() else os.path.abspath(image_path)

    seed = int(param_node.evalParm('seed'))
    ss_steps = int(param_node.evalParm('ss_steps'))
    ss_strength = float(param_node.evalParm('ss_strength'))
    slat_steps = int(param_node.evalParm('slat_steps'))
    slat_strength = float(param_node.evalParm('slat_strength'))
    mesh_simplify = float(param_node.evalParm('mesh_simplify'))
    texture_size = int(param_node.evalParm('texture_size'))

    poll_interval = 2.0
    timeout = 600.0

    if not os.path.exists(image_path):
        print(f"Error: Input image file does not exist at: {image_path}")
        work_item.setFailed()
        return

    payload = {
        "image_path": image_path,
        "seed": seed,
        "ss_sampling_steps": ss_steps,
        "ss_guidance_strength": ss_strength,
        "slat_sampling_steps": slat_steps,
        "slat_guidance_strength": slat_strength,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size
    }

    print(f"Triggering TRELLIS generation on: {api_url}")
    generate_endpoint = f"{api_url.rstrip('/')}/generate"

    try:
        response = requests.post(generate_endpoint, json=payload, timeout=10)
        response.raise_for_status()

        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"Task queued successfully. Task ID: {task_id}")

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
"""

    core_node.parm("script").set(enhanced_cook_script)
    core_node.parm("inprocess").set(True)

    # 4. Create the digital asset definition from the Subnet Container Node
    hda_node_type_name = f"houtrellis_{VERSION}"
    hda_label = f"HouTrellis TOP {VERSION.upper()}"

    new_hda_node = hda_node.createDigitalAsset(
        name=hda_node_type_name,
        hda_file_name=hda_path,
        description=hda_label,
        min_num_inputs=0,
        max_num_inputs=1,
    )

    # Save parameters and definition to ensure they are exposed on the HDA type interface
    hda_definition = new_hda_node.type().definition()
    hda_definition.setParmTemplateGroup(group)
    hda_definition.save(hda_path)

    print(f"SUCCESS: Created Digital Asset successfully!")
    print(f"HDA File Location: {hda_path}")
    print(f"HDA Type Name: {hda_node_type_name}")

    # Cleanup temporary nodes
    topnet.destroy()


if __name__ == "__main__":
    build_trellis_top_hda()
