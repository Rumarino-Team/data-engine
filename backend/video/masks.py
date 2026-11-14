from __future__ import annotations
from typing import Any, TYPE_CHECKING
from uuid import uuid4
from functools import lru_cache
from pathlib import Path
import logging, re

if TYPE_CHECKING:
    from core.state import SessionState
    from schemas.video import InteractiveMaskRLE, MaskOverride
import numpy as np
from core.errors import OperationError
from video.mask_encoding import encode_mask_payload, encode_live_mask_payload

logger = logging.getLogger(__name__)

def as_2d_bool(mask: Any) -> np.ndarray | None:
    """The mask as a 2D bool array, or None when it is not one (after dropping singleton axes)."""
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.ndim != 2:
        mask_array = np.squeeze(mask_array)
    return mask_array if mask_array.ndim == 2 else None

def manifest_frame_payload(frame_masks: dict[int, np.ndarray]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for obj_id, mask in frame_masks.items():
        mask_array = as_2d_bool(mask)
        if mask_array is not None:
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


def _overrides_dir(session_path: Path | None) -> Path:
    if session_path is None:
        raise ValueError("No active session for mask overrides.")
    return session_path / "masks" / "overrides"


def store_mask_override(session_path: Path | None, frame_idx: int, obj_id: int, mask: np.ndarray) -> "MaskOverride":
    """Publish pixels as a new immutable artifact; the caller accepts its reference."""
    from schemas.video import MaskOverride
    from utils import write_mask_manifest
    artifact = f"{uuid4().hex}.json"
    write_mask_manifest(_overrides_dir(session_path) / artifact, encode_override(frame_idx, obj_id, mask).model_dump())
    return MaskOverride(frame_idx=frame_idx, obj_id=obj_id, artifact=artifact)


@lru_cache(maxsize=256)
def _read_override(path: Path) -> tuple["InteractiveMaskRLE", dict[str, Any]]:
    # Artifacts are immutable, so the decoded RLE and its wire payload can be shared.
    from schemas.video import InteractiveMaskRLE
    from utils import load_mask_manifest
    rle = InteractiveMaskRLE.model_validate(load_mask_manifest(path))
    return rle, encode_live_mask_payload(decode_override(rle))


def _load_override(ref: "MaskOverride", session_path: Path | None) -> tuple["InteractiveMaskRLE", dict[str, Any]]:
    path = _overrides_dir(session_path) / ref.artifact
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


def override_pixels(ref: "MaskOverride", session_path: Path | None) -> np.ndarray:
    return decode_override(_load_override(ref, session_path)[0])


def override_payload(ref: "MaskOverride", session_path: Path | None) -> dict[str, Any]:
    return _load_override(ref, session_path)[1]


def retain_interactive_masks(session: SessionState, frame_idx: int, obj_ids: list[int], encoded_masks: list[dict]) -> None:
    from sessions.metadata import current_masks_dir
    from utils import write_mask_manifest
    masks_dir = current_masks_dir(session)
    if masks_dir is None:
        raise ValueError("No active session for interactive masks.")
    payload = {
        str(obj_id): {"input_revision": session.annotations.input_revisions.get(obj_id, 0),
                      "mask": mask}
        for obj_id, mask in zip(obj_ids, encoded_masks) if obj_id in session.annotations.objects
    }
    # Publish an immutable artifact first, then accept its reference with annotations.
    # If metadata acceptance fails, the previous reference still names untouched pixels.
    artifact_name = f"{uuid4().hex}.json"
    write_mask_manifest(masks_dir / "interactive" / artifact_name, payload)
    session.interactive_mask_files[str(frame_idx)] = artifact_name


# Interactive mask artifacts are named by a uuid; session.json is hand-editable, so a reference is never a path.
INTERACTIVE_ARTIFACT_NAME = re.compile(r"[0-9a-f]{32}\.json")


def validate_interactive_mask_references(references: Any) -> dict[str, str]:
    """The frame -> artifact map of a saved session, or ValueError when any entry could name another file."""
    if not isinstance(references, dict):
        raise ValueError("Interactive mask references must be an object.")
    for frame, name in references.items():
        if not str(frame).isdigit() or not isinstance(name, str) or not INTERACTIVE_ARTIFACT_NAME.fullmatch(name):
            raise ValueError(f"Invalid interactive mask reference for frame {frame}.")
    return references


@lru_cache(maxsize=1)
def _read_interactive_mask_file(path: Path) -> dict[str, Any]:
    # UUID artifacts are immutable. Keep their cache separate from the propagation manifest.
    from utils import load_mask_manifest
    return load_mask_manifest(path)


def load_interactive_masks(session: SessionState, frame_idx: int) -> dict[str, Any]:
    from sessions.metadata import current_masks_dir
    artifact_name = session.interactive_mask_files.get(str(frame_idx))
    if artifact_name is None:
        return {}
    if not isinstance(artifact_name, str) or not INTERACTIVE_ARTIFACT_NAME.fullmatch(artifact_name):
        raise ValueError("Invalid interactive mask artifact reference.")
    masks_dir = current_masks_dir(session)
    if masks_dir is None:
        raise ValueError("No active session for interactive masks.")
    return _read_interactive_mask_file(masks_dir / "interactive" / artifact_name)


def effective_mask_data(session: SessionState, frame_idx: int) -> dict[str, Any]:
    """One authoritative frame: explicit input > interactive result > propagation.

    Empty masks are present values, so they suppress lower-precedence sources.
    Generated results expire on edits to their object's inputs, not display labels.
    """
    from video.manifest_cache import load_cached_mask_manifest
    objects, sources = {}, {}
    manifest_path = Path(session.mask_manifest_path) if session.mask_manifest_path else None
    frame = {}
    if manifest_path and manifest_path.exists():
        frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
    for obj_id, mask in frame.get("objects", {}).items():
        object_id = int(obj_id)
        if (object_id in session.annotations.objects and
                frame.get("input_revisions", {}).get(str(obj_id)) == session.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = mask
            sources[str(obj_id)] = "manifest"
    for obj_id, result in load_interactive_masks(session, frame_idx).items():
        object_id = int(obj_id)
        if (object_id in session.annotations.objects and
                result["input_revision"] == session.annotations.input_revisions.get(object_id, 0)):
            objects[str(obj_id)] = result["mask"]
            sources[str(obj_id)] = "live"
    for mask in session.annotations.mask_overrides:
        if mask.frame_idx == frame_idx:
            objects[str(mask.obj_id)] = override_payload(mask, session.active_session_dir)
            sources[str(mask.obj_id)] = "override"
    return {"session_id": session.active_session_id, "annotation_revision": session.annotations.revision,
            "frame_idx": frame_idx, "objects": objects, "sources": sources}


def mask_overlay_path(session: SessionState, frame_idx: int) -> Path:
    """The propagation overlay image for a frame, only while it still shows the frame's effective masks."""
    from sessions.metadata import current_masks_dir
    from sessions.paths import path_is_relative_to
    from video.manifest_cache import load_cached_mask_manifest
    masks_dir = current_masks_dir(session)
    manifest_path = Path(session.mask_manifest_path) if session.mask_manifest_path else None
    if masks_dir is None or not manifest_path or not manifest_path.exists():
        raise OperationError(404, "Mask frame not available")
    frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
    effective = effective_mask_data(session, frame_idx)
    if (not frame.get("overlay_path") or effective["objects"] != frame.get("objects") or
            any(source != "manifest" for source in effective["sources"].values())):
        raise OperationError(404, "Mask overlay is obsolete")
    file_path = (masks_dir / frame["overlay_path"]).resolve()
    if not path_is_relative_to(file_path, masks_dir.resolve()) or not file_path.is_file():
        raise OperationError(404, "Mask frame not available")
    return file_path
