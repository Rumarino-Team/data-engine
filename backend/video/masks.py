from typing import Any, TYPE_CHECKING
from uuid import uuid4
from functools import lru_cache
from pathlib import Path
import logging, re

if TYPE_CHECKING:
    from schemas.video import InteractiveMaskRLE, MaskOverride
import numpy as np
from video.mask_encoding import encode_mask_payload, encode_live_mask_payload

logger = logging.getLogger(__name__)

def manifest_frame_payload(frame_masks: dict[int, np.ndarray]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for obj_id, mask in frame_masks.items():
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.ndim != 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            continue
        objects[str(int(obj_id))] = encode_mask_payload(mask_array)
    return {"objects": objects}

def mask_logits_to_2d_bool(mask_logits: Any) -> np.ndarray:
    mask_array = (mask_logits > 0.0).detach().cpu().numpy()
    mask_array = np.squeeze(mask_array).astype(bool)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = np.squeeze(mask_array, axis=0)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask_array.shape}")
    return mask_array

def serialize_video_segments_for_response(
    video_segments: dict,
    *,
    max_frames: int,
    max_mask_values: int,
) -> tuple[dict, bool, int, int]:
    """Serialize masks to JSON-safe payload with optional size limits."""
    serialized: dict[int, dict[int, list]] = {}
    total_mask_values = 0
    returned_frames = 0

    for frame_idx, obj_dict in sorted(video_segments.items(), key=lambda item: int(item[0])):
        if max_frames >= 0 and returned_frames >= max_frames:
            break

        frame_masks: dict[int, np.ndarray] = {}
        frame_mask_values = 0
        for obj_id, mask in obj_dict.items():
            mask_array = np.asarray(mask)
            frame_mask_values += int(mask_array.size)
            frame_masks[int(obj_id)] = mask_array

        if max_mask_values >= 0 and (total_mask_values + frame_mask_values) > max_mask_values:
            break

        serialized[int(frame_idx)] = {
            obj_id: mask_array.tolist()
            for obj_id, mask_array in frame_masks.items()
        }
        total_mask_values += frame_mask_values
        returned_frames += 1

    truncated = returned_frames < len(video_segments)
    return serialized, truncated, returned_frames, total_mask_values


def decode_override(mask: "InteractiveMaskRLE") -> np.ndarray:
    return np.repeat(np.arange(len(mask.counts)) % 2, mask.counts).astype(bool).reshape(mask.height, mask.width)


def encode_override(frame_idx: int, obj_id: int, mask: np.ndarray) -> "InteractiveMaskRLE":
    from schemas.video import InteractiveMaskRLE
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return InteractiveMaskRLE(frame_idx=frame_idx, obj_id=obj_id,
                              height=mask.shape[0], width=mask.shape[1], counts=counts)


def _overrides_dir() -> Path:
    from sessions.metadata import current_masks_dir
    masks_dir = current_masks_dir()
    if masks_dir is None:
        raise ValueError("No active session for mask overrides.")
    return masks_dir / "overrides"


def store_mask_override(frame_idx: int, obj_id: int, mask: np.ndarray) -> "MaskOverride":
    """Publish pixels as a new immutable artifact; the caller accepts its reference."""
    from schemas.video import MaskOverride
    from utils import write_mask_manifest
    artifact = f"{uuid4().hex}.json"
    write_mask_manifest(_overrides_dir() / artifact, encode_override(frame_idx, obj_id, mask).model_dump())
    return MaskOverride(frame_idx=frame_idx, obj_id=obj_id, artifact=artifact)


@lru_cache(maxsize=256)
def _read_override(path: Path) -> tuple["InteractiveMaskRLE", dict[str, Any]]:
    # Artifacts are immutable, so the decoded RLE and its wire payload can be shared.
    from schemas.video import InteractiveMaskRLE
    from utils import load_mask_manifest
    rle = InteractiveMaskRLE.model_validate(load_mask_manifest(path))
    return rle, encode_live_mask_payload(decode_override(rle))


def _load_override(ref: "MaskOverride", session_path: Path | None = None) -> tuple["InteractiveMaskRLE", dict[str, Any]]:
    path = (session_path / "masks" / "overrides" if session_path else _overrides_dir()) / ref.artifact
    try:
        # The cache trusts immutability, not existence; a reclaimed file must not stay readable.
        if not path.is_file():
            raise FileNotFoundError(path)
        rle, payload = _read_override(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Mask override artifact is unavailable: {ref.artifact}") from error
    if (rle.frame_idx, rle.obj_id) != (ref.frame_idx, ref.obj_id):
        raise ValueError(f"Mask override artifact does not match its reference: {ref.artifact}")
    return rle, payload


def override_pixels(ref: "MaskOverride", *, session_path: Path | None = None) -> np.ndarray:
    return decode_override(_load_override(ref, session_path)[0])


def override_payload(ref: "MaskOverride") -> dict[str, Any]:
    return _load_override(ref)[1]


def reclaim_override_artifacts(keep: set[str], *, session_path: Path | None = None) -> None:
    """Best effort: an unreferenced artifact is garbage once metadata no longer names it."""
    try:
        overrides_dir = session_path / "masks" / "overrides" if session_path else _overrides_dir()
        if not overrides_dir.is_dir():
            return
        for path in overrides_dir.glob("*.json"):
            if path.name not in keep:
                path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to reclaim mask override artifacts", exc_info=True)


def discard_override_artifacts(names: set[str], *, session_path: Path | None = None) -> None:
    for name in names:
        try:
            directory = session_path / "masks" / "overrides" if session_path else _overrides_dir()
            (directory / name).unlink(missing_ok=True)
        except (OSError, ValueError):
            logger.warning("Unable to remove mask override artifact %s", name, exc_info=True)


def retain_interactive_masks(frame_idx: int, obj_ids: list[int], encoded_masks: list[dict]) -> None:
    from core.state import state
    from sessions.metadata import current_masks_dir
    from utils import write_mask_manifest
    masks_dir = current_masks_dir()
    if masks_dir is None:
        raise ValueError("No active session for interactive masks.")
    payload = {
        str(obj_id): {"input_revision": state.annotations.input_revisions.get(obj_id, 0),
                      "mask": mask}
        for obj_id, mask in zip(obj_ids, encoded_masks) if obj_id in state.annotations.objects
    }
    # Publish an immutable artifact first, then accept its reference with annotations.
    # If metadata acceptance fails, the previous reference still names untouched pixels.
    artifact_name = f"{uuid4().hex}.json"
    write_mask_manifest(masks_dir / "interactive" / artifact_name, payload)
    state.interactive_mask_files[str(frame_idx)] = artifact_name


@lru_cache(maxsize=1)
def _read_interactive_mask_file(path: Path) -> dict[str, Any]:
    # UUID artifacts are immutable. Keep their cache separate from the propagation manifest.
    from utils import load_mask_manifest
    return load_mask_manifest(path)


def load_interactive_masks(frame_idx: int) -> dict[str, Any]:
    from core.state import state
    from sessions.metadata import current_masks_dir
    artifact_name = state.interactive_mask_files.get(str(frame_idx))
    if artifact_name is None:
        return {}
    if not isinstance(artifact_name, str) or not re.fullmatch(r"[0-9a-f]{32}\.json", artifact_name):
        raise ValueError("Invalid interactive mask artifact reference.")
    masks_dir = current_masks_dir()
    if masks_dir is None:
        raise ValueError("No active session for interactive masks.")
    return _read_interactive_mask_file(masks_dir / "interactive" / artifact_name)


def effective_mask_data(frame_idx: int) -> dict[str, Any]:
    """One authoritative frame: explicit input > interactive result > propagation.

    Empty masks are present values, so they suppress lower-precedence sources.
    Generated results expire on edits to their object's inputs, not display labels.
    """
    from core.state import state
    from sessions.metadata import current_masks_dir
    from video.manifest_cache import load_cached_mask_manifest
    objects, sources = {}, {}
    masks_dir = current_masks_dir()
    manifest_path = Path(state.mask_manifest_path) if state.mask_manifest_path else None
    frame = {}
    if manifest_path and manifest_path.exists():
        frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
    for obj_id, mask in frame.get("objects", {}).items():
        object_id = int(obj_id)
        if (object_id in state.annotations.objects and
                frame.get("input_revisions", {}).get(str(obj_id)) == state.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = mask
            sources[str(obj_id)] = "manifest"
    for obj_id, result in load_interactive_masks(frame_idx).items():
        object_id = int(obj_id)
        if (object_id in state.annotations.objects and
                result["input_revision"] == state.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = result["mask"]
            sources[str(obj_id)] = "live"
    for mask in state.annotations.mask_overrides:
        if mask.frame_idx == frame_idx:
            objects[str(mask.obj_id)] = override_payload(mask)
            sources[str(mask.obj_id)] = "override"
    return {"session_id": state.active_session_id, "annotation_revision": state.annotations.revision,
            "state_epoch": state.video_state_epoch, "frame_idx": frame_idx,
            "objects": objects, "sources": sources}
