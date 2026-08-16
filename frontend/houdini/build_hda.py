import os
import sys

# Define HDA version suffix here
VERSION = "v05"


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

    # Create internal nodes
    server_node = hda_node.createNode("genericgenerator", "start_server")
    log_start_node = hda_node.createNode("pythonscript", "start_logging")
    trigger_node = hda_node.createNode("urlrequest", "trigger_generation")
    log_stop_node = hda_node.createNode("pythonscript", "stop_logging")
    poll_node = hda_node.createNode("pythonscript", "poll_status")

    # Wire the internal nodes sequentially
    subnet_input = hda_node.node("subnetinput1")
    subnet_output = hda_node.node("subnetoutput1")

    if subnet_input and subnet_output:
        print(
            "Wiring internal nodes including auto-server startup and background logging..."
        )
        server_node.setInput(0, subnet_input)
        log_start_node.setInput(0, server_node)
        trigger_node.setInput(0, log_start_node)
        log_stop_node.setInput(0, trigger_node)
        poll_node.setInput(0, log_stop_node)
        subnet_output.setInput(0, poll_node)

        # Position them nicely inside the HDA graph viewport
        subnet_input.setPosition(hou.Vector2(0, 5))
        server_node.setPosition(hou.Vector2(0, 3))
        log_start_node.setPosition(hou.Vector2(0, 1))
        trigger_node.setPosition(hou.Vector2(0, -1))
        log_stop_node.setPosition(hou.Vector2(0, -3))
        poll_node.setPosition(hou.Vector2(0, -5))
        subnet_output.setPosition(hou.Vector2(0, -7))

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
    shell_command = """# Unset Houdini python environment variables to prevent virtualenv import errors
unset PYTHONHOME
unset PYTHONPATH

# Check if server is already running on port 8000
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs | grep -q "200"; then
    echo "=== HouTrellis server is already running. Proceeding... ==="
else
    echo "=== Server not detected. Starting HouTrellis FastAPI Server... ==="
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

    # 4. Configure the start_logging parameters
    # This node generates the client-side task ID and launches the background log tailer thread
    start_logging_script = """import uuid
import os
import threading
import time
import hou

def start_task_logging(work_item):
    # 1. Generate client-side task ID
    task_id = str(uuid.uuid4())
    work_item.setAttrib('task_id', task_id)

    # 2. Create the empty log file
    log_dir = "/home/admin/houtrellis/backend/outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_id}.log")
    with open(log_file, "w") as f:
         f.write("=== Monitoring HouTrellis GPU Job ===\\n")

    # 3. Initialize custom hou.logging Source
    log_source_name = f"HouTrellis_{task_id}"
    logger = hou.logging.createSource(log_source_name)
    print(f"Registered HouTrellis logging source: {log_source_name}")

    # Global state to share shutdown flag with background thread
    if not hasattr(hou, '_trellis_threads'):
        hou._trellis_threads = {}

    stop_event = threading.Event()
    hou._trellis_threads[task_id] = stop_event

    # 4. Define log tailer thread
    def tail_file(file_path, stop_evt, log_src):
        try:
            with open(file_path, "r") as f:
                # Seek to end
                f.seek(0, 2)
                while not stop_evt.is_set():
                    line = f.readline()
                    if line:
                        # Log directly into Houdini's log view
                        hou.logging.log(line.strip(), source_name=log_src, severity=hou.loggingSeverity.Message)
                    else:
                        time.sleep(0.5)
        except Exception as e:
            print(f"Log tailer exception: {e}")

    # Launch background thread
    t = threading.Thread(target=tail_file, args=(log_file, stop_event, log_source_name), daemon=True)
    t.start()

start_task_logging(work_item)
"""
    log_start_node.parm("script").set(start_logging_script)
    log_start_node.parm("inprocess").set(True)

    # 5. Configure parameters on the internal Trigger URL Request Node
    # httptype: 1 corresponds to POST
    trigger_node.parm("httptype").set(1)
    trigger_node.parm("baseurl").set('`chs("../api_url")`/generate')
    trigger_node.parm("usecontenttype").set(1)
    trigger_node.parm("contenttype").set("application/json")
    trigger_node.parm("payloadtype").set(3)

    # Python-based payload expression incorporating the pre-generated 'task_id'
    python_payload_expression = """import json
payload_data = {
  "task_id": work_item.attribValue("task_id"), # Send the pre-generated log task ID
  "image_path": hou.evalParm("../image_path"),
  "seed": hou.evalParm("../seed"),
  "ss_sampling_steps": hou.evalParm("../ss_steps"),
  "ss_guidance_strength": hou.evalParm("../ss_strength"),
  "slat_sampling_steps": hou.evalParm("../slat_steps"),
  "slat_guidance_strength": hou.evalParm("../slat_strength"),
  "mesh_simplify": hou.evalParm("../mesh_simplify"),
  "texture_size": hou.evalParm("../texture_size")
}
return json.dumps(payload_data)"""
    trigger_node.parm("payloadcustom").setExpression(
        python_payload_expression, hou.exprLanguage.Python
    )

    # Save Response directly to status_data attribute
    trigger_node.parm("saveto").set(1)
    trigger_node.parm("attributename").set("status_data")

    # 6. Configure parameters on the internal stop_logging parameters
    # This node cleans up and destroys the log source and background thread
    stop_task_logging_script = """import hou

def stop_task_logging(work_item):
    task_id = work_item.attribValue('task_id')
    log_source_name = f"HouTrellis_{task_id}"

    # Signal the log tailer thread to exit
    if hasattr(hou, '_trellis_threads') and task_id in hou._trellis_threads:
        print(f"Stopping log tailer thread for: {task_id}")
        hou._trellis_threads[task_id].set()
        del hou._trellis_threads[task_id]

    # Unregister the logging source
    try:
        hou.logging.destroySource(log_source_name)
        print(f"Unregistered HouTrellis logging source: {log_source_name}")
    except Exception as e:
        print(f"Error unregistering logging source: {e}")

stop_task_logging(work_item)
"""
    log_stop_node.parm("script").set(stop_task_logging_script)
    log_stop_node.parm("inprocess").set(True)

    # 7. Configure parameters on the internal Poll Node (Python Script TOP)
    polling_script = """# Clean, Focused Status Polling
import os
import time
import requests
import json
import pdg

def cook_status_polling(work_item):
    node = work_item.holder.node
    param_node = node.parent() if (node.parent() and node.parent().type().name() != "topnet") else node
    api_url = param_node.evalParm('api_url')

    task_id = work_item.attribValue('task_id')
    status_endpoint = f"{api_url.rstrip('/')}/status/{task_id}"

    poll_interval = 2.0
    timeout = 600.0
    start_time = time.time()

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
                # Let the background logging thread do the work. Just sleep and poll status.
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

    # 8. Create the digital asset definition from the Subnet Container Node
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
