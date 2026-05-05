import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CACHE_ROOT = BACKEND_ROOT / "cache"
GENERATED_FRAMES_ROOT = CACHE_ROOT / "frames"
WINDOW_FRAMES_ROOT = CACHE_ROOT / "windows"
SESSION_CACHE_ROOT = CACHE_ROOT / "sessions"
SAVED_ROOT = BACKEND_ROOT / "saved"
DEFAULT_PROMPT_TRACK_BATCH_SIZE = int(os.getenv("TRACK_PROMPT_BATCH_SIZE", "32"))
DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD = int(os.getenv("TRACK_STREAMING_FRAME_THRESHOLD", "256"))
DEFAULT_MAX_INTERACTIVE_LIVE_MASKS = int(os.getenv("VIDEO_SAVE_MAX_LIVE_MASKS", "5000"))
