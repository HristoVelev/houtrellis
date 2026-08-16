import os
import sys

# Define HDA version suffix here
VERSION = "v03"


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

    # Create the internal Generic Generator TOP node to auto-start the FastAPI server if needed
    server_node = hda_node.createNode("genericgenerator", "start_server")

    # Create the internal URL Request TOP (httptype POST)
    trigger_node = hda_node.createNode("urlrequest", "trigger_generation")

    # Create the internal Python Script TOP to handle clean, focused status polling
    poll_node = hda_node.createNode("pythonscript", "poll_status")

    # Wire the internal nodes between Subnet Input and Subnet Output
    subnet_input = hda_node.node("subnetinput1")
    subnet_output = hda_node.node("subnetoutput1")

    if subnet_input and subnet_output:
        print("Wiring internal nodes including auto-server startup...")
        server_node.setInput(0, subnet_input)
        trigger_node.setInput(0, server_node)
        poll_node.setInput(0, trigger_node)
        subnet_output.setInput(0, poll_node)

        # Position them nicely inside the HDA graph viewport
        subnet_input.setPosition(hou.Vector2(0, 4))
        server_node.setPosition(hou.Vector2(0, 2))
        trigger_node.setPosition(hou.Vector2(0, 0))
        poll_node.setPosition(hou.Vector2(0, -2))
        subnet_output.setPosition(hou.Vector2(0, -4))

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

    # 3. Configure the start_server Shell TOP parameters
    shell_command = """# Check if server is already running on port 8000
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs | grep -q "200"; then
    echo "=== HouTrellis server is already running. Proceeding... ==="
else
    echo "=== Server not detected. Starting HouTrellis FastAPI Server... ==="
    # Start the server in the background, redirecting stdout/stderr and detaching
    nohup /home/admin/houtrellis/backend/venv/bin/python /home/admin/houtrellis/backend/app.py > /home/admin/houtrellis/backend/server_output.log 2>&1 &

    # Wait a few seconds for Uvicorn to initialize and bind to port 8000
    echo "Waiting for server to initialize..."
    for i in {1..15}; do
        if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs | grep -q "200"; then
            echo "=== Server successfully initialized! ==="
            break
        fi
        sleep 1
    done
fi
"""
    server_node.parm("shellcommand").set(1)
    server_node.parm("pdg_command").set(shell_command)

    # 4. Configure parameters on the internal Trigger URL Request Node
    # httptype: 1 corresponds to POST
    trigger_node.parm("httptype").set(1)

    # Set the target generate URL dynamically from HDA api_url parameter
    trigger_node.parm("baseurl").set('`chs("../api_url")`/generate')

    # Configure JSON payload
    trigger_node.parm("usecontenttype").set(1)
    trigger_node.parm("contenttype").set("application/json")

    # payloadtype: 3 corresponds to Custom String
    trigger_node.parm("payloadtype").set(3)

    # Embed raw JSON body that references our parent HDA parameters using Houdini backticks
    raw_payload_json = """{
  "image_path": "`chs("../image_path")`",
  "seed": `chi("../seed")`,
  "ss_sampling_steps": `chi("../ss_steps")`,
  "ss_guidance_strength": `ch("../ss_strength")`,
  "slat_sampling_steps": `chi("../slat_steps")`,
  "slat_guidance_strength": `ch("../slat_strength")`,
  "mesh_simplify": `ch("../mesh_simplify")`,
  "texture_size": `chi("../texture_size")`
}"""
    trigger_node.parm("payloadcustom").set(raw_payload_json)

    # Save Response directly to a PDG attribute
    # saveto: 1 corresponds to Attribute
    trigger_node.parm("saveto").set(1)
    trigger_node.parm("attributename").set("status_data")

    # 5. Configure parameters on the internal Poll Node (Python Script TOP)
    polling_script = """# Clean, Focused Status Polling
import os
import time
import requests
import pdg

def cook_status_polling(work_item):
    node = work_item.holder.node
    # Find the HDA subnet node dynamically
    param_node = node.parent() if (node.parent() and node.parent().type().name() != "topnet") else node
    api_url = param_node.evalParm('api_url')

    # Extract task ID from our previous urlrequest response attribute
    status_data = work_item.attribValue('status_data')
    if not status_data or 'task_id' not in status_data:
        print("Error: Could not find valid 'task_id' in work item attributes.")
        work_item.setFailed()
        return

    task_id = status_data['task_id']
    status_endpoint = f"{api_url.rstrip('/')}/status/{task_id}"

    poll_interval = 2.0
    timeout = 600.0
    start_time = time.time()

    print(f"Starting status polling for Task ID: {task_id}")
    while True:
        if time.time() - start_time > timeout:
            print(f"Error: Polling timed out after {timeout} seconds.")
            work_item.setFailed()
            return

        try:
            res = requests.get(status_endpoint, timeout=5).json()
            status = res["status"]

            if status == "completed":
                glb_path = res["output_path"]
                print(f"Success! Model generated at: {glb_path}")
                work_item.addResultData(glb_path, "file/glb", 0)
                break
            elif status == "failed":
                print(f"Error: Generation failed on backend with error: {res.get('error')}")
                work_item.setFailed()
                return
            elif status in ["queued", "processing"]:
                print(f"State: {status}... sleeping for {poll_interval}s")
                time.sleep(poll_interval)
            else:
                print(f"Error: Unexpected status value: {status}")
                work_item.setFailed()
                return
        except Exception as e:
            print(f"Exception during polling: {e}")
            work_item.setFailed()
            return

# Run the polling loop
cook_status_polling(work_item)
"""
    poll_node.parm("script").set(polling_script)
    poll_node.parm("inprocess").set(True)

    # 6. Create the digital asset definition from the Subnet Container Node
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
