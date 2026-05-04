from typing import Any
import numpy as np
from utils import encode_mask_to_rle, mask_bbox_xywh

def manifest_frame_payload(frame_masks: dict[int, np.ndarray]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for obj_id, mask in frame_masks.items():
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.ndim != 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            continue
        objects[str(int(obj_id))] = {
            "size": [int(mask_array.shape[0]), int(mask_array.shape[1])],
            "rle": encode_mask_to_rle(mask_array),
            "bbox": mask_bbox_xywh(mask_array),
        }
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

