from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.video import InteractiveMaskRLE
import numpy as np
from video.mask_encoding import encode_mask_payload, encode_live_mask_payload

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


def accepted_mask(frame_idx: int, obj_id: int, mask: np.ndarray) -> "InteractiveMaskRLE":
    from schemas.video import InteractiveMaskRLE
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return InteractiveMaskRLE(frame_idx=frame_idx, obj_id=obj_id,
                              height=mask.shape[0], width=mask.shape[1], counts=counts)


def retain_interactive_masks(frame_idx: int, obj_ids: list[int], encoded_masks: list[dict]) -> None:
    from core.state import state
    state.interactive_masks[str(frame_idx)] = {
        str(obj_id): {"input_revision": state.annotations.input_revisions.get(obj_id, 0),
                      "mask": mask}
        for obj_id, mask in zip(obj_ids, encoded_masks) if obj_id in state.annotations.objects
    }


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
    manifest_path = masks_dir / "manifest.json" if masks_dir else None
    frame = {}
    if manifest_path and manifest_path.exists():
        frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
    for obj_id, mask in frame.get("objects", {}).items():
        object_id = int(obj_id)
        if (object_id in state.annotations.objects and
                frame.get("input_revisions", {}).get(str(obj_id)) == state.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = mask
            sources[str(obj_id)] = "manifest"
    for obj_id, result in state.interactive_masks.get(str(frame_idx), {}).items():
        object_id = int(obj_id)
        if (object_id in state.annotations.objects and
                result["input_revision"] == state.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = result["mask"]
            sources[str(obj_id)] = "live"
    for mask in state.annotations.mask_overrides:
        if mask.frame_idx == frame_idx:
            objects[str(mask.obj_id)] = encode_live_mask_payload(decode_override(mask))
            sources[str(mask.obj_id)] = "override"
    return {"session_id": state.active_session_id, "annotation_revision": state.annotations.revision,
            "state_epoch": state.video_state_epoch, "frame_idx": frame_idx,
            "objects": objects, "sources": sources}
