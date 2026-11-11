import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CACHE_ROOT = BACKEND_ROOT / "cache"
SESSION_CACHE_ROOT = CACHE_ROOT / "sessions"
SAVED_ROOT = BACKEND_ROOT / "saved"
# Model sources are pinned to the commits the results were produced with. CoTracker is code fetched from GitHub
# and run, so it must never follow a moving branch; SAM2 weights come from Hugging Face at a fixed revision.
# Changing either is a deliberate update: check tracks and masks against the old ones first.
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
SAM2_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
COTRACKER_REPO = "facebookresearch/co-tracker"
COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
DEFAULT_PROMPT_TRACK_BATCH_SIZE = int(os.getenv("TRACK_PROMPT_BATCH_SIZE", "32"))
DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD = int(os.getenv("TRACK_STREAMING_FRAME_THRESHOLD", "256"))
