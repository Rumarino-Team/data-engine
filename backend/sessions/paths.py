import os
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse
from fastapi import HTTPException
from core.config import IMAGE_EXTENSIONS, PROJECT_ROOT, VIDEO_EXTENSIONS

def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def normalize_input_path(path_value: str) -> str:
    normalized = str(path_value).strip().strip('"').strip("'")
    if not normalized:
        return normalized

    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            normalized = parsed.path or ""
            if parsed.netloc and parsed.netloc != "localhost":
                normalized = f"//{parsed.netloc}{normalized}"

            if os.name == "nt" and normalized.startswith("/") and len(normalized) > 2 and normalized[2] == ":":
                normalized = normalized[1:]

    normalized = unquote(normalized)
    normalized = normalized.replace("\\ ", " ")
    return normalized

def resolve_input_path(path_value: str, expect_dir: Optional[bool] = None) -> Path:
    normalized = normalize_input_path(path_value)
    if not normalized:
        raise HTTPException(status_code=400, detail="Path cannot be empty.")

    expanded = Path(os.path.expandvars(os.path.expanduser(normalized)))

    if expanded.is_absolute():
        candidate_paths = [expanded.resolve()]
    else:
        cwd_candidate = (Path.cwd() / expanded).resolve()
        project_candidate = (PROJECT_ROOT / expanded).resolve()
        candidate_paths = [cwd_candidate]
        if project_candidate != cwd_candidate:
            candidate_paths.append(project_candidate)

    resolved_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    if not resolved_path.exists():
        tried_paths = ", ".join(str(path) for path in candidate_paths)
        raise HTTPException(
            status_code=404,
            detail=f"Path not found: '{normalized}'. Tried: {tried_paths}"
        )

    if expect_dir is True and not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Expected a directory path, got file: {resolved_path}")

    if expect_dir is False and not resolved_path.is_file():
        raise HTTPException(status_code=400, detail=f"Expected a file path, got directory: {resolved_path}")

    return resolved_path

def validate_video_input_path(resolved_input_path: Path) -> None:
    if not resolved_input_path.is_file():
        return
    suffix = resolved_input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return
    if suffix in IMAGE_EXTENSIONS:
        detail = (
            f"Expected a frames directory or video file, got a single image file: {resolved_input_path}. "
            "Provide a directory containing image frames."
        )
    else:
        detail = (
            f"Unsupported input file type: {resolved_input_path.suffix or '<none>'}. "
            "Provide a directory of image frames or a video file (.mp4, .mov, .avi, .mkv, .webm, .m4v)."
        )
    raise HTTPException(status_code=400, detail=detail)

def resolve_saved_session_layout(resolved_input_path: Path) -> Optional[tuple[Path, Path, Path]]:
    if not resolved_input_path.is_dir():
        return None
    session_json = resolved_input_path / "session.json"
    if not session_json.exists():
        return None
    frames_dir = resolved_input_path / "frames"
    masks_dir = resolved_input_path / "masks"
    if not frames_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Saved session directory is missing required frames/ folder: {frames_dir}",
        )
    if not masks_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Saved session directory is missing required masks/ folder: {masks_dir}",
        )
    return resolved_input_path.resolve(), frames_dir.resolve(), masks_dir.resolve()

