import json
import os
import sys
import time

import requests
import yaml

API_URL = "http://127.0.0.1:8000/api/v1"


def parse_and_run_pipeline(yaml_path):
    print(f"=== Reading Declarative Pipeline: {yaml_path} ===")

    if not os.path.exists(yaml_path):
        print(f"Error: YAML configuration file not found at: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    steps = config.get("pipeline", [])
    print(f"Loaded {len(steps)} pipeline steps.")

    # Global session storage to hold output variables for dynamic chaining
    # E.g. {"generate_concept_art.image_path": "/home/admin/houtrellis/backend/outputs/...png"}
    session_variables = {}

    for index, step in enumerate(steps, start=1):
        step_name = step.get("step")
        endpoint = step.get("endpoint")
        payload = step.get("payload", {})

        print(f"\n[{index}/{len(steps)}] Executing Step: '{step_name}' ({endpoint})")

        # 1. Resolve dynamic variables in payload (e.g. $[generate_concept_art.image_path])
        resolved_payload = resolve_payload_variables(payload, session_variables)

        # 2. Trigger the task
        trigger_url = f"{API_URL}/{endpoint}"
        print(f"Triggering request: {trigger_url}")
        try:
            response = requests.post(trigger_url, json=resolved_payload)
            response.raise_for_status()
        except Exception as e:
            print(f"Error: Trigger request failed: {e}")
            if response is not None:
                print(f"Response: {response.text}")
            sys.exit(1)

        task_data = response.json()
        task_id = task_data["task_id"]
        log_file = task_data.get("log_path")
        print(f"Task successfully scheduled! Task ID: {task_id}")

        # 3. Periodically poll task status and tail the local task log file
        status_url = f"{API_URL}/{endpoint}/status/{task_id}"
        poll_interval = 2.0
        start_time = time.time()
        timeout = 600.0  # 10 minutes max

        last_logged_line = 0
        print("Tailing task log output:")

        while True:
            if time.time() - start_time > timeout:
                print(f"Error: Step '{step_name}' timed out after {timeout} seconds.")
                sys.exit(1)

            try:
                status_res = requests.get(status_url)
                status_res.raise_for_status()
                status_data = status_res.json()
                status = status_data["status"]
            except Exception as se:
                print(f"Error getting status: {se}")
                sys.exit(1)

            # Tail the log file if it exists on disk
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, "r") as lf:
                        lines = lf.readlines()
                        if len(lines) > last_logged_line:
                            for line in lines[last_logged_line:]:
                                print(f"  > {line.strip()}")
                            last_logged_line = len(lines)
                except Exception:
                    pass

            if status == "completed":
                print(f"Success! Step '{step_name}' completed successfully.")

                # Store all output fields in our global session variables for downstream chaining!
                for k, v in status_data.items():
                    if v is not None:
                        session_variables[f"{step_name}.{k}"] = v
                break

            elif status == "failed":
                print(
                    f"Error: Step '{step_name}' failed with error: {status_data.get('error')}"
                )
                sys.exit(1)

            time.sleep(poll_interval)

    print("\n==================================================")
    print("🎉 Declarative Pipeline Execution Completed Successfully!")
    print("Final Output variables generated:")
    for k, v in session_variables.items():
        print(f"  - {k} -> {v}")
    print("==================================================")


def resolve_payload_variables(payload, session_vars):
    """Recursively replaces any $[step.variable] string in the payload with its session value."""
    if isinstance(payload, dict):
        resolved = {}
        for k, v in payload.items():
            resolved[k] = resolve_payload_variables(v, session_vars)
        return resolved
    elif isinstance(payload, list):
        return [resolve_payload_variables(item, session_vars) for item in payload]
    elif isinstance(payload, str):
        if payload.startswith("$[") and payload.endswith("]"):
            var_name = payload[2:-1]
            if var_name in session_vars:
                print(
                    f"  - Resolved variable '{payload}' to '{session_vars[var_name]}'"
                )
                return session_vars[var_name]
            else:
                print(
                    f"Error: Reference variable '{payload}' not found in previous step outputs!"
                )
                sys.exit(1)
        return payload
    else:
        return payload


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "test_pipeline.yml"
    parse_and_run_pipeline(yaml_path)
