"""RLE for saved masks, adaptive RLE/packed bits for live responses."""

import base64
from typing import Any
import numpy as np


def encode_mask_payload(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("Expected a nonempty 2D mask.")
    height, width = mask.shape
    flat = mask.reshape(-1)
    # Find transitions in NumPy instead of looping over every pixel in Python.
    padded = np.empty(flat.size + 2, dtype=bool)
    padded[0] = padded[-1] = False
    padded[1:-1] = flat
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = boundaries[::2], boundaries[1::2]
    runs = np.column_stack((starts, ends - starts)).tolist()
    if runs:
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        bbox = [int(columns[0]), int(rows[0]),
                int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)]
    else:
        bbox = [0, 0, 0, 0]
    return {"size": [int(height), int(width)], "rle": runs, "bbox": bbox}


def encode_live_mask_payload(mask: np.ndarray) -> dict[str, Any]:
    """Use MSB-first packed bits when run pairs would cost more than base64 bytes.

    Twelve bytes per JSON run is a conservative estimate, not a timing guarantee.
    Small masks keep RLE to avoid fixed packed-payload overhead.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("Expected a nonempty 2D mask.")
    flat = mask.reshape(-1)
    run_count = int(flat[0]) + int(np.count_nonzero(flat[1:] & ~flat[:-1]))
    base64_size = 4 * (((flat.size + 7) // 8 + 2) // 3)
    if run_count <= max(32, base64_size // 12):
        return encode_mask_payload(mask)
    data = np.packbits(flat, bitorder="big").tobytes()
    return {"size": [int(mask.shape[0]), int(mask.shape[1])],
            "encoding": "packed-bits", "data": base64.b64encode(data).decode("ascii")}
