import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
# Frames a folder may hold. SAM2 only reads JPEG, so the other formats are converted to JPEG when a video is loaded.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
JPEG_QUALITY = 95
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
# How many accepted changes can be undone. The history only lives in the backend's memory.
UNDO_LIMIT = int(os.getenv("UNDO_HISTORY_LIMIT", "10"))
# Points tracked per CoTracker pass. Every pass runs the whole video, and the number of points hardly changes a pass's
# time or GPU memory (300 points: 29 s in one pass against 178 s in passes of 32, same 2.6 GB peak), so a pass takes many.
# A pass that runs out of memory is retried with half as many points.
DEFAULT_PROMPT_TRACK_BATCH_SIZE = int(os.getenv("TRACK_PROMPT_BATCH_SIZE", "256"))
# Videos with at least this many frames are tracked by streaming them from disk; 0 (the default) streams every video.
# Streaming was as fast as tracking in memory at every length measured and used about half the GPU memory.
DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD = int(os.getenv("TRACK_STREAMING_FRAME_THRESHOLD", "0"))
