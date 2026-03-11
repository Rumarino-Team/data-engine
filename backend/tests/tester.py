import base64
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json
import zipfile

import requests
import numpy as np

# Configuration
BASE_URL = os.getenv("DATA_ENGINE_BASE_URL", "http://127.0.0.1:8000")
BEDROOM_ZIP_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

BEDROOM_DIR = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_BEDROOM_DIR", str(PROJECT_ROOT / "bedroom"))
        )
    )
)
VIDEO_DIR = BEDROOM_DIR  # Will use bedroom frames for video tests
TRACKING_VIDEO_PATH = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_TRACKING_VIDEO", str(PROJECT_ROOT / "apple.mp4"))
        )
    )
)
API_FILE = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_API_FILE", str(BACKEND_DIR / "api.py"))
        )
    )
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


ONLINE_MODE = _env_bool("DATA_ENGINE_ONLINE_MODE", True)
ONLINE_BATCH_SIZE = int(os.getenv("DATA_ENGINE_BATCH_SIZE", "32"))
OFFLOAD_VIDEO_TO_CPU = _env_bool("DATA_ENGINE_OFFLOAD_VIDEO_TO_CPU", True)
OFFLOAD_STATE_TO_CPU = _env_bool("DATA_ENGINE_OFFLOAD_STATE_TO_CPU", False)


def download_and_extract_bedroom():
    """Downloads and extracts the bedroom video frames if not already present."""
    if BEDROOM_DIR.exists() and BEDROOM_DIR.is_dir():
        # Check if directory has files
        if any(BEDROOM_DIR.iterdir()):
            print(f"Bedroom directory already exists with files. Skipping download.")
            return True
    
    print(f"Downloading bedroom.zip from {BEDROOM_ZIP_URL}...")
    zip_path = PROJECT_ROOT / "bedroom.zip"
    
    try:
        # Download the file with progress indication
        response = requests.get(BEDROOM_ZIP_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with zip_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
        
        print("\nDownload complete!")
        
        # Extract the zip file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_ROOT)
        
        print(f"Extraction complete!")
        
        # Clean up the zip file
        zip_path.unlink(missing_ok=True)
        print(f"Cleaned up {zip_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError downloading or extracting bedroom.zip: {e}")
        # Clean up partial downloads
        zip_path.unlink(missing_ok=True)
        return False


def wait_for_server(url, timeout=30):
    """Waits for the FastAPI server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Server failed to start within the timeout period.")
    return False


def init_video_state(video_dir):
    """Calls the /video/init_state endpoint."""
    url = f"{BASE_URL}/video/init_state"
    payload = {
        "video_frames_dir": str(video_dir),
        "online_mode": ONLINE_MODE,
        "batch_size": ONLINE_BATCH_SIZE,
        "offload_video_to_cpu": OFFLOAD_VIDEO_TO_CPU,
        "offload_state_to_cpu": OFFLOAD_STATE_TO_CPU,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"Video state initialized successfully for '{video_dir}'.")
    else:
        print(f"Failed to initialize video state. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200


def reset_video_state():
    """Calls the /video/reset_state endpoint."""
    url = f"{BASE_URL}/video/reset_state"
    response = requests.post(url)
    if response.status_code == 200:
        print("Video state reset successfully.")
    else:
        print(f"Failed to reset video state. Status: {response.status_code}, Response: {response.text}")


def add_new_points_or_box(frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
    """Calls the /video/add_new_points_or_box endpoint."""
    url = f"{BASE_URL}/video/add_new_points_or_box"
    payload = {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "clear_old_points": clear_old_points
    }
    if points is not None:
        payload["points"] = points.tolist() if isinstance(points, np.ndarray) else points
    if labels is not None:
        payload["labels"] = labels.tolist() if isinstance(labels, np.ndarray) else labels
    if box is not None:
        payload["box"] = box.tolist() if isinstance(box, np.ndarray) else box
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Added points/box for frame {frame_idx}, object {obj_id}.")
        print(f"  Object IDs: {response_data.get('out_obj_ids', [])}")
    else:
        print(f"Failed to add points/box. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def propagate_in_video(start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
    """Calls the /video/propagate_in_video endpoint."""
    url = f"{BASE_URL}/video/propagate_in_video"
    payload = {
        "start_frame_idx": start_frame_idx,
        "max_frame_num_to_track": max_frame_num_to_track,
        "reverse": reverse,
        "online_mode": ONLINE_MODE,
        "batch_size": ONLINE_BATCH_SIZE,
        "include_masks_in_response": False,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        online_mode = response_data.get("online_mode")
        batch_size = response_data.get("batch_size")
        if online_mode is not None:
            print(f"Online batching: {'enabled' if online_mode else 'disabled'} (batch_size={batch_size})")
        num_frames = response_data.get("video_segments_total_frames", len(response_data.get("video_segments", {})))
        saved_paths = response_data.get("saved_mask_paths", {})
        print(f"Propagation successful! Processed {num_frames} frames.")
        returned_frames = response_data.get("video_segments_returned_frames", len(response_data.get("video_segments", {})))
        if returned_frames:
            print(f"Returned {returned_frames} frame masks in API response.")
        print(f"Saved masks for {len(saved_paths)} frames.")
        for frame_idx, paths in saved_paths.items():
            print(f"  Frame {frame_idx}: {paths}")
    else:
        print(f"Failed to propagate. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def stop_server_process(server_process):
    """Stops FastAPI server process and its child processes reliably."""
    if server_process is None:
        return

    if server_process.poll() is not None:
        print("Server already stopped.")
        return

    try:
        if os.name != "nt":
            os.killpg(server_process.pid, signal.SIGTERM)
        else:
            server_process.terminate()
    except ProcessLookupError:
        print("Server process not found during shutdown.")
        return

    try:
        server_process.wait(timeout=10)
        print("Server shut down successfully.")
        return
    except subprocess.TimeoutExpired:
        print("Server did not terminate in time, forcing kill.")
    except KeyboardInterrupt:
        print("Interrupted during shutdown, forcing kill.")

    try:
        if os.name != "nt":
            os.killpg(server_process.pid, signal.SIGKILL)
        else:
            server_process.kill()
    except ProcessLookupError:
        pass

    try:
        server_process.wait(timeout=5)
    except Exception:
        pass

    print("Server killed.")


def run_video_tests():
    """Runs the video masking test suite."""
    print("\n" + "=" * 60)
    print("RUNNING VIDEO MASKING TESTS")
    print("=" * 60)
    
    # Check if video directory exists
    if not VIDEO_DIR.exists():
        print(f"\nVideo directory '{VIDEO_DIR}' not found. Skipping video tests.")
        return
    
    print(f"\n--- Running test: video_masking ---")
    
    # Initialize video state
    if not init_video_state(VIDEO_DIR):
        print("Failed to initialize video state. Skipping video tests.")
        return
    
    # Test case from user request
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    
    # Let's add a positive click at (x, y) = (210, 350) to get started
    print(f"\nAdding first point at (210, 350) on frame {ann_frame_idx}...")
    points = np.array([[210, 350]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    success, _ = add_new_points_or_box(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )
    
    if not success:
        print("Failed to add first point. Aborting video test.")
        return
    
    time.sleep(0.5)
    
    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`
    print(f"\nAdding second point at (250, 220) on frame {ann_frame_idx}...")
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    success, _ = add_new_points_or_box(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )
    
    if not success:
        print("Failed to add second point. Aborting video test.")
        return
    
    time.sleep(0.5)
    
    # Propagate through the video
    print(f"\nPropagating masks through video...")
    success, result = propagate_in_video()
    
    if success:
        print("\n✓ Video masking test completed successfully!")
    else:
        print("\n✗ Video masking test failed.")
    
    # Reset video state for cleanup
    reset_video_state()


def load_tracking_video(video_path):
    """Calls the /tracking/load_video endpoint."""
    url = f"{BASE_URL}/tracking/load_video"
    response = requests.post(url, json={"video_path": str(video_path)})
    if response.status_code == 200:
        response_data = response.json()
        print(f"Video loaded successfully: {video_path}")
        print(f"  Shape: {response_data.get('shape', [])}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
    else:
        print(f"Failed to load tracking video. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def track_grid(grid_size=15, add_support_grid=True):
    """Calls the /tracking/track_grid endpoint."""
    url = f"{BASE_URL}/tracking/track_grid"
    payload = {
        "grid_size": grid_size,
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Grid tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track grid. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def track_points(queries, add_support_grid=True):
    """Calls the /tracking/track_points endpoint."""
    url = f"{BASE_URL}/tracking/track_points"
    payload = {
        "queries": queries if isinstance(queries, list) else queries.tolist(),
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Point tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track points. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def run_tracking_tests():
    """Runs the tracking test suite."""
    print("\n" + "=" * 60)
    print("RUNNING TRACKING TESTS")
    print("=" * 60)
    
    # Check if tracking video exists
    if not TRACKING_VIDEO_PATH.exists():
        print(f"\nTracking video '{TRACKING_VIDEO_PATH}' not found. Skipping tracking tests.")
        return
    
    # Test 1: Load video
    print(f"\n--- Test 1: Load tracking video ---")
    success, result = load_tracking_video(TRACKING_VIDEO_PATH)
    
    if not success:
        print("Failed to load tracking video. Skipping remaining tracking tests.")
        return
    
    time.sleep(1)
    
    # Test 2: Track grid
    print(f"\n--- Test 2: Track grid of points ---")
    success, result = track_grid(grid_size=10, add_support_grid=True)
    
    if success:
        print("\n✓ Grid tracking test completed successfully!")
    else:
        print("\n✗ Grid tracking test failed.")
    
    time.sleep(1)
    
    # Test 3: Track specific points
    print(f"\n--- Test 3: Track specific query points ---")
    # Define some query points: [frame, x, y]
    # Let's track a few points starting from frame 0
    queries = [
        [0, 400, 350],  # Center point
        [10, 600, 500],  # Upper-left area
        [20, 750, 600],  # Lower-right area
        [30, 900, 200]
    ]
    success, result = track_points(queries, add_support_grid=False)
    
    if success:
        print("\n✓ Point tracking test completed successfully!")
    else:
        print("\n✗ Point tracking test failed.")
    
    time.sleep(1)
    
    # Test 4: Track points with support grid
    print(f"\n--- Test 4: Track points with support grid ---")
    queries = [
        [0, 320, 240],  # Single point with support grid
    ]
    success, result = track_points(queries, add_support_grid=True)
    
    if success:
        print("\n✓ Point tracking with support grid completed successfully!")
    else:
        print("\n✗ Point tracking with support grid failed.")
    
    print("\n" + "=" * 60)
    print("TRACKING TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Download and extract bedroom video frames
    print("=" * 60)
    print("SETTING UP TEST DATA")
    print("=" * 60)
    if not download_and_extract_bedroom():
        print("Failed to download bedroom data. Exiting.")
        exit(1)

    if not API_FILE.exists():
        print(f"API file not found: {API_FILE}")
        exit(1)
    
    # Start the FastAPI server as a background process
    server_process = subprocess.Popen(
        [sys.executable, "-m", "fastapi", "dev", str(API_FILE)],
        cwd=str(BACKEND_DIR),
        start_new_session=(os.name != "nt"),
    )
    print(f"\nStarting FastAPI server with PID: {server_process.pid}...")

    try:
        # Wait for the server to be ready
        # The root endpoint in api.py returns a simple message
        if wait_for_server(f"{BASE_URL}/", timeout=30):
            # Run the video tests
            run_video_tests()
            
            # Run the tracking tests
            run_tracking_tests()
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED")
            print("=" * 60)
        else:
            print("Could not connect to the server. Aborting tests.")

    finally:
        # Stop the server
        print("\nShutting down the server...")
        stop_server_process(server_process)
