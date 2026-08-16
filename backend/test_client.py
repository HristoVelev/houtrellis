import json
import os
import sys
import time

import requests

API_URL = "http://127.0.0.1:8000"


def test_backend_generation():
    print("=== Testing HouTrellis Backend Standalone ===")

    # 1. Check if server is running
    try:
        # Just hit a dummy endpoint or check if port is open
        requests.get(API_URL)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to HouTrellis server at {API_URL}.")
        print("Please make sure you start the server first by running:")
        print("   python backend/app.py")
        sys.exit(1)

    # 2. Check for custom input image in command line arguments
    custom_image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if custom_image_path:
        test_image_path = custom_image_path
        if not os.path.exists(test_image_path):
            print(f"Error: Custom image path not found: {test_image_path}")
            sys.exit(1)
        print(f"Using custom input image: {test_image_path}")
        is_mock_image = False
    else:
        test_image_path = "backend/test_input.png"
        is_mock_image = True
        if not os.path.exists(test_image_path):
            with open(test_image_path, "wb") as f:
                # Writing 100 bytes of dummy data as a mock image
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 92)
            print(f"Created a mock input image at: {test_image_path}")

    # 3. Trigger generate request
    payload = {
        "image_path": os.path.abspath(test_image_path),
        "seed": 42,
        "ss_sampling_steps": 12,
        "ss_guidance_strength": 7.5,
        "slat_sampling_steps": 12,
        "slat_guidance_strength": 3.0,
    }

    print("\nSending generation request...")
    response = requests.post(f"{API_URL}/generate", json=payload)
    if response.status_code != 200:
        print(f"Generation request failed! Response: {response.text}")
        sys.exit(1)

    task_data = response.json()
    task_id = task_data["task_id"]
    print(f"Generation started successfully! Task ID: {task_id}")

    # 4. Poll status
    print("\nPolling server status...")
    timeout = 300
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            print(f"Test timed out after {timeout} seconds!")
            sys.exit(1)

        status_response = requests.get(f"{API_URL}/status/{task_id}")
        status_data = status_response.json()
        status = status_data["status"]

        print(f"Current Status: {status}")

        if status == "completed":
            print(
                f"\nSuccess! Generated file is located at: {status_data['output_path']}"
            )
            break
        elif status == "failed":
            print(f"\nGeneration failed with error: {status_data.get('error')}")
            sys.exit(1)

        time.sleep(1)

    # Clean up test input image only if it was a mock
    if is_mock_image and os.path.exists(test_image_path):
        os.remove(test_image_path)
        print("\nCleaned up mock input image.")


if __name__ == "__main__":
    test_backend_generation()
