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
import shutil

import requests
import numpy as np

# Configuration
BASE_URL = os.getenv("DATA_ENGINE_BASE_URL", "http://127.0.0.1:8000")
BEDROOM_ZIP_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip"
APPLE_VIDEO_URL = os.getenv(
    "DATA_ENGINE_APPLE_VIDEO_URL",
    "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4",
)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
TEST_BEDROOM_DIR = SCRIPT_DIR / "bedroom"
TEST_APPLE_DIR = SCRIPT_DIR / "apple"

BEDROOM_DIR = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_BEDROOM_DIR", str(TEST_BEDROOM_DIR))
        )
    )
)
VIDEO_DIR = BEDROOM_DIR  # Will use bedroom frames for video tests
APPLE_SOURCE_VIDEO_PATH = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_APPLE_SOURCE_VIDEO", str(PROJECT_ROOT / "apple.mp4"))
        )
    )
)
TRACKING_VIDEO_PATH = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("DATA_ENGINE_TRACKING_VIDEO", str(TEST_APPLE_DIR / "apple.mp4"))
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
JOB_POLL_INTERVAL_SECONDS = float(os.getenv("DATA_ENGINE_JOB_POLL_INTERVAL", "0.5"))
JOB_TIMEOUT_SECONDS = float(os.getenv("DATA_ENGINE_JOB_TIMEOUT", "1800"))


def _print_job_progress(job):
    stage_label = job.get("stage_label") or job.get("stage") or "Working"
    progress = job.get("progress")
    current = job.get("current")
    total = job.get("total")
    window_index = job.get("window_index")
    window_count = job.get("window_count")
    frame_idx = job.get("frame_idx")
    history = job.get("stage_history") or []
    message = job.get("message") or ""

    parts = [stage_label]
    if window_index is not None and window_count:
        parts.append(f"Window {window_index}/{window_count}")
    if frame_idx is not None:
        parts.append(f"Frame {frame_idx}")
    if isinstance(progress, (int, float)):
        parts.append(f"{progress * 100:5.1f}%")
    if current is not None and total:
        parts.append(f"{current}/{total}")
    if message:
        parts.append(message)
    if history:
        latest_history_message = history[-1].get("message")
        if latest_history_message and latest_history_message != message:
            parts.append(f"last: {latest_history_message}")

    print("\r  " + " | ".join(parts), end="", flush=True)


def wait_for_job(job_start_response, timeout=JOB_TIMEOUT_SECONDS):
    """Polls a background job until it completes and returns its result payload."""
    job_id = job_start_response.get("job_id")
    if not job_id:
        print(f"Invalid job start response: {job_start_response}")
        return False, None

    started = time.time()
    last_status = None
    while time.time() - started < timeout:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        if response.status_code != 200:
            print(f"\nFailed to poll job {job_id}. Status: {response.status_code}, Response: {response.text}")
            return False, None

        job = response.json().get("job", {})
        status = job.get("status")
        if status != last_status:
            print(f"\n  Job {job_id}: {status}")
            last_status = status
        _print_job_progress(job)

        if status == "completed":
            print()
            return True, job.get("result")
        if status == "failed":
            print()
            error = job.get("error") or {}
            print(f"Job failed: {error.get('message', 'Unknown error')}")
            if error.get("detail"):
                print(f"  Detail: {error.get('detail')}")
            return False, job

        time.sleep(JOB_POLL_INTERVAL_SECONDS)

    print(f"\nTimed out waiting for job {job_id} after {timeout:.1f}s.")
    return False, None


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_apple_test_video() -> bool:
    """Ensures apple.mp4 is available inside backend/tests/apple so outputs stay there."""
    ensure_clean_dir(TEST_APPLE_DIR)
    if TRACKING_VIDEO_PATH.exists():
        return True
    if APPLE_SOURCE_VIDEO_PATH.exists():
        try:
            shutil.copy2(APPLE_SOURCE_VIDEO_PATH, TRACKING_VIDEO_PATH)
            print(f"Copied apple test video to: {TRACKING_VIDEO_PATH}")
            return True
        except Exception as error:
            print(f"Failed to prepare apple test video from local source: {error}")
            return False

    print(f"Apple source video not found: {APPLE_SOURCE_VIDEO_PATH}")
    print(f"Downloading apple.mp4 from {APPLE_VIDEO_URL}...")
    temp_path = TRACKING_VIDEO_PATH.with_suffix(".mp4.tmp")
    try:
        response = requests.get(APPLE_VIDEO_URL, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rApple download progress: {progress:.1f}%", end="", flush=True)
        temp_path.replace(TRACKING_VIDEO_PATH)
        print(f"\nDownloaded apple test video to: {TRACKING_VIDEO_PATH}")
        return True
    except Exception as error:
        print(f"\nFailed to download apple test video: {error}")
        temp_path.unlink(missing_ok=True)
        return False


def write_json_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def copy_output_into_dir(output_video_path: str, output_dir: Path) -> str:
    if not output_video_path:
        return output_video_path
    source_path = Path(output_video_path)
    if not source_path.exists():
        return output_video_path
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / source_path.name
    if source_path.resolve() == target_path.resolve():
        return str(target_path)
    if target_path.exists():
        target_path.unlink()
    shutil.move(str(source_path), str(target_path))
    return str(target_path)


def download_and_extract_bedroom():
    """Downloads and extracts the bedroom video frames if not already present."""
    if BEDROOM_DIR.exists() and BEDROOM_DIR.is_dir():
        # Check if directory has files
        if any(BEDROOM_DIR.iterdir()):
            print(f"Bedroom directory already exists with files. Skipping download.")
            return True
    
    print(f"Downloading bedroom.zip from {BEDROOM_ZIP_URL}...")
    zip_path = SCRIPT_DIR / "bedroom.zip"
    extract_root = SCRIPT_DIR
    
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
            zip_ref.extractall(extract_root)
        
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
        print(f"Video init job started for '{video_dir}'.")
        success, result = wait_for_job(response.json())
        if success:
            print(f"Video state initialized successfully for '{video_dir}'.")
        return success
    else:
        print(f"Failed to initialize video state. Status: {response.status_code}, Response: {response.text}")
    return False


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
        print("Propagation job started.")
        success, response_data = wait_for_job(response.json())
        if not success or response_data is None:
            return False, response_data
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
        return True, response_data
    else:
        print(f"Failed to propagate. Status: {response.status_code}, Response: {response.text}")
    return False, None


def track_prompt_points(add_support_grid=False):
    """Runs prompt-point tracking from current SAM prompt state and fetches the disk-backed result."""
    url = f"{BASE_URL}/tracking/track_prompt_points"
    response = requests.post(url, json={"add_support_grid": bool(add_support_grid)})
    if response.status_code != 200:
        print(f"Failed to start prompt tracking. Status: {response.status_code}, Response: {response.text}")
        return False, None

    print("Prompt tracking job started.")
    success, job_result = wait_for_job(response.json())
    if not success or not isinstance(job_result, dict):
        return False, job_result

    result_id = job_result.get("tracking_result_id")
    if not result_id:
        print(f"Prompt tracking job did not return tracking_result_id: {job_result}")
        return False, job_result

    result_response = requests.get(f"{BASE_URL}/tracking/results/{result_id}")
    if result_response.status_code != 200:
        print(
            f"Failed to fetch tracking result {result_id}. "
            f"Status: {result_response.status_code}, Response: {result_response.text}"
        )
        return False, job_result

    tracking_payload = result_response.json().get("result", {})
    summary = {
        "job_result": job_result,
        "result_id": result_id,
        "num_points": tracking_payload.get("num_points"),
        "num_frames": tracking_payload.get("num_frames"),
        "tracking_mode": tracking_payload.get("tracking_mode"),
        "first_point": (tracking_payload.get("points") or [{}])[0],
    }
    output_path = BEDROOM_DIR / f"prompt_tracking_{result_id}.json"
    write_json_output(output_path, summary)
    print(f"Prompt tracking result fetched and summarized at: {output_path}")
    return True, tracking_payload


def restore_bedroom_masks_output(result_payload):
    """Copy propagated masks back to BEDROOM_DIR/masks to preserve legacy test output location."""
    if not isinstance(result_payload, dict):
        return False

    manifest_path_raw = result_payload.get("mask_manifest_path") or result_payload.get("state.mask_manifest_path")
    if not manifest_path_raw:
        print("No mask manifest path returned; cannot restore bedroom mask output folder.")
        return False

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.exists():
        print(f"Mask manifest not found: {manifest_path}")
        return False

    source_masks_dir = manifest_path.parent
    target_masks_dir = BEDROOM_DIR / "masks"

    try:
        if target_masks_dir.exists():
            shutil.rmtree(target_masks_dir)
        shutil.copytree(source_masks_dir, target_masks_dir)
        print(f"Restored bedroom masks output to: {target_masks_dir}")
        return True
    except Exception as error:
        print(f"Failed to restore bedroom masks output: {error}")
        return False


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
        restore_bedroom_masks_output(result)
        print("\n✓ Video masking test completed successfully!")
    else:
        print("\n✗ Video masking test failed.")
        reset_video_state()
        return

    print(f"\nRunning prompt-point tracking from bedroom prompts...")
    success, tracking_result = track_prompt_points(add_support_grid=False)
    if success:
        print(
            "\n✓ Bedroom prompt tracking test passed "
            f"({tracking_result.get('num_points', 0)} pts × {tracking_result.get('num_frames', 0)} frames)"
        )
    else:
        print("\n✗ Bedroom prompt tracking test failed.")
    
    # Reset video state for cleanup
    reset_video_state()


def load_tracking_video(video_path, model_name=None):
    """Calls the /tracking/load_video endpoint."""
    url = f"{BASE_URL}/tracking/load_video"
    payload = {"video_path": str(video_path)}
    if model_name is not None:
        payload["model_name"] = model_name
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Video loaded successfully: {video_path}")
        print(f"  Model: {response_data.get('model_name', 'N/A')}")
        print(f"  Shape: {response_data.get('shape', [])}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
    else:
        print(f"Failed to load tracking video. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def _ensure_mode_labeled_output_path(output_video_path: str, expected_mode: str | None) -> str:
    """Ensures the output filename includes the expected mode label when requested."""
    if not output_video_path:
        return output_video_path
    if expected_mode is None:
        return output_video_path

    output_path = Path(output_video_path)
    name = output_path.name
    opposite_mode = "online" if expected_mode == "offline" else "offline"

    if f"_{expected_mode}_" in name:
        return output_video_path

    if f"_{opposite_mode}_" in name:
        print(
            f"Warning: Output file appears labeled as {opposite_mode}: {output_video_path}. "
            f"Leaving filename unchanged."
        )
        return output_video_path

    stem = output_path.stem
    suffix = output_path.suffix

    # Insert mode before trailing timestamp if present: ..._<ts>.mp4
    prefix, sep, maybe_ts = stem.rpartition("_")
    if sep and maybe_ts.isdigit() and prefix:
        new_stem = f"{prefix}_{expected_mode}_{maybe_ts}"
    else:
        new_stem = f"{stem}_{expected_mode}"

    new_path = output_path.with_name(f"{new_stem}{suffix}")

    if output_path.exists() and new_path != output_path:
        output_path.rename(new_path)
        print(f"Renamed output file to include mode label: {new_path}")
        return str(new_path)

    return output_video_path


def track_grid(grid_size=15, add_support_grid=True, expected_mode=None, output_dir=None):
    """Calls the /tracking/track_grid endpoint."""
    url = f"{BASE_URL}/tracking/track_grid"
    payload = {
        "grid_size": grid_size,
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        if expected_mode:
            response_data["output_video_path"] = _ensure_mode_labeled_output_path(
                response_data.get("output_video_path", ""), expected_mode
            )
        if output_dir:
            response_data["output_video_path"] = copy_output_into_dir(
                response_data.get("output_video_path", ""), Path(output_dir)
            )
        print(f"Grid tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track grid. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def track_points(queries, add_support_grid=True, expected_mode=None, output_dir=None):
    """Calls the /tracking/track_points endpoint."""
    url = f"{BASE_URL}/tracking/track_points"
    payload = {
        "queries": queries if isinstance(queries, list) else queries.tolist(),
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        if expected_mode:
            response_data["output_video_path"] = _ensure_mode_labeled_output_path(
                response_data.get("output_video_path", ""), expected_mode
            )
        if output_dir:
            response_data["output_video_path"] = copy_output_into_dir(
                response_data.get("output_video_path", ""), Path(output_dir)
            )
        print(f"Point tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track points. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def run_apple_tracking_tests():
    """Runs apple.mp4 CoTracker tests using the current backend-supported model."""
    generated_outputs = []

    print("\n" + "=" * 60)
    print("RUNNING APPLE TRACKING TESTS")
    print("=" * 60)

    if not ensure_apple_test_video():
        return

    print(f"\n--- Apple Test 1: Load tracking video ---")
    success, result = load_tracking_video(TRACKING_VIDEO_PATH)

    if not success:
        print("Failed to load apple tracking video. Skipping remaining tests.")
        return

    time.sleep(1)

    print(f"\n--- Apple Test 2: Track grid of points ---")
    success, result = track_grid(grid_size=10, add_support_grid=True, output_dir=TEST_APPLE_DIR)

    if success:
        num_points = result.get("num_points", 0)
        num_frames = result.get("num_frames", 0)
        if result.get("output_video_path"):
            generated_outputs.append(result.get("output_video_path"))
        print(f"\n✓ Apple grid tracking test passed  ({num_points} pts × {num_frames} frames)")
    else:
        print(f"\n✗ Apple grid tracking test failed.")

    time.sleep(1)

    print(f"\n--- Apple Test 3: Restored legacy point tracking ---")
    queries = [
        [0, 400, 350],
        [10, 600, 500],
        [20, 750, 600],
        [30, 900, 200],
    ]
    success, result = track_points(queries, add_support_grid=False, output_dir=TEST_APPLE_DIR)

    if success:
        num_points = result.get("num_points", 0)
        num_frames = result.get("num_frames", 0)
        if result.get("output_video_path"):
            generated_outputs.append(result.get("output_video_path"))
        write_json_output(
            TEST_APPLE_DIR / "apple_legacy_points_summary.json",
            {
                "queries": queries,
                "num_points": num_points,
                "num_frames": num_frames,
                "output_video_path": result.get("output_video_path"),
            },
        )
        print(f"\n✓ Apple legacy point tracking test passed  ({num_points} pts × {num_frames} frames)")
    else:
        print(f"\n✗ Apple legacy point tracking test failed.")

    time.sleep(1)

    print(f"\n--- Apple Test 4: Track point with support grid ---")
    queries = [
        [0, 320, 240],
    ]
    success, result = track_points(queries, add_support_grid=True, output_dir=TEST_APPLE_DIR)

    if success:
        num_points = result.get("num_points", 0)
        num_frames = result.get("num_frames", 0)
        if result.get("output_video_path"):
            generated_outputs.append(result.get("output_video_path"))
        print(f"\n✓ Apple point tracking (+ support grid) test passed  ({num_points} pts × {num_frames} frames)")
    else:
        print(f"\n✗ Apple point tracking (+ support grid) test failed.")

    if generated_outputs:
        print(f"\nApple generated outputs:")
        for output_path in generated_outputs:
            print(f"  - {output_path}")


def run_tracking_tests():
    """Runs the current tracking integration suite."""
    run_apple_tracking_tests()


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
