import numpy as np
import cv2
from pathlib import Path
import shutil
import json
from typing import Any


def _color_from_obj_id(obj_id):
    """Deterministic BGR color derived from object id."""
    object_id = int(obj_id)
    return np.array(
        [
            (37 * object_id + 79) % 256,
            (67 * object_id + 131) % 256,
            (97 * object_id + 191) % 256,
        ],
        dtype=np.uint8,
    )


def show_mask(image, mask, random_color=False, borders=True, color=None):
    if color is not None:
        color = np.asarray(color, dtype=np.uint8)
    elif random_color:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
    else:
        color = np.array([255, 144, 30], dtype=np.uint8)  # BGR for blue

    if mask is None:
        return image

    mask = np.asarray(mask)
    if mask.size == 0:
        return image
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask after squeeze, got shape {mask.shape}"
        )

    h, w = mask.shape[-2:]
    mask_bool = mask.astype(bool)

    # Create a colored mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask_bool] = color

    # Nothing to blend on this frame/object.
    if not np.any(mask_bool):
        return image

    # Blend the colored mask with the original image
    # Use NumPy blending (safe for any valid selection size).
    image_pixels = image[mask_bool].astype(np.float32)
    mask_pixels = color_mask[mask_bool].astype(np.float32)
    image[mask_bool] = np.clip(0.5 * image_pixels + 0.5 * mask_pixels, 0, 255).astype(np.uint8)

    if borders:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
    return image


def prepare_video_masks_output(video_dir, masks_dir=None, *, clear_existing=True):
    """
    Prepare output directory and frame file list for streaming mask writes.

    Returns:
    - frame_files: sorted list of frame paths
    - masks_dir: output directory path
    """
    video_path = Path(video_dir)
    masks_dir = Path(masks_dir) if masks_dir is not None else video_path / "masks"

    if clear_existing and masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(exist_ok=True)

    frame_files = sorted([
        f for f in video_path.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])

    return frame_files, masks_dir


def save_single_video_mask_frame(frame_files, masks_dir, frame_idx, obj_masks):
    """
    Save one propagated mask frame overlay to disk.

    Returns:
    - str path to saved file, or None when frame index is out of range / unreadable.
    """
    if frame_idx < 0 or frame_idx >= len(frame_files):
        return None

    frame_path = frame_files[frame_idx]
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None

    output_frame = frame.copy()
    for obj_id, mask in obj_masks.items():
        output_frame = show_mask(
            output_frame,
            mask,
            random_color=False,
            borders=True,
            color=_color_from_obj_id(obj_id),
        )

    output_filename = f"frame_{frame_idx:05d}_masks.png"
    output_path = masks_dir / output_filename
    cv2.imwrite(str(output_path), output_frame)
    return str(output_path)


def build_empty_mask_manifest(
    *,
    source_video_path: str | None,
    resolved_video_frames_dir: str,
    num_frames: int,
    frame_height: int | None,
    frame_width: int | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "source_video_path": source_video_path,
        "resolved_video_frames_dir": resolved_video_frames_dir,
        "num_frames": int(num_frames),
        "frame_height": int(frame_height) if frame_height is not None else None,
        "frame_width": int(frame_width) if frame_width is not None else None,
        "frames": {},
    }


def write_mask_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, separators=(",", ":"))


def load_mask_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
