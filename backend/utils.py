import numpy as np
import cv2
from pathlib import Path
import shutil
import hashlib


def show_mask(image, mask, random_color=False, borders=True):
    if random_color:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
    else:
        color = np.array([255, 144, 30], dtype=np.uint8)  # BGR for blue

    mask = np.asarray(mask)
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

    # Blend the colored mask with the original image
    # The alpha channel is simulated by weighting
    image[mask_bool] = cv2.addWeighted(image[mask_bool], 0.5, color_mask[mask_bool], 0.5, 0)

    if borders:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
    return image


def show_points(image, coords, labels, marker_size=20):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for p in pos_points:
        cv2.drawMarker(image, (int(p[0]), int(p[1])), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    for p in neg_points:
        cv2.drawMarker(image, (int(p[0]), int(p[1])), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    return image


def show_box(image, boxes):
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return image


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Render mask overlay images using OpenCV and return them as byte arrays.

    Parameters:
    - image: numpy array (H,W,3) RGB image to display under masks
    - masks: iterable of binary numpy arrays (H,W) for each mask
    - scores: iterable of floats corresponding to each mask
    - point_coords, box_coords, input_labels: optional annotations to render
    - borders: whether to draw contours around masks

    Returns: list of byte arrays, each containing a PNG-encoded image
    """
    output_images = []
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Create a fresh copy for each mask's image
        output_image = image_bgr.copy()

        # Apply mask
        output_image = show_mask(
            output_image, mask, random_color=True, borders=borders)

        # Draw points and boxes if they exist
        if point_coords is not None:
            assert input_labels is not None
            output_image = show_points(
                output_image, point_coords, input_labels)
        if box_coords is not None:
            output_image = show_box(output_image, box_coords)

        # Add score text
        if len(scores) > 1:
            score_array = np.asarray(score)
            if score_array.size == 0:
                score_value = float("nan")
            else:
                score_value = float(score_array.reshape(-1)[0])
            text = f"Mask {i+1}, Score: {score_value:.3f}"
            cv2.putText(output_image, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the image to a byte array
        _, buffer = cv2.imencode('.png', output_image)
        output_images.append(buffer.tobytes())

    return output_images


def save_video_masks(video_dir, video_segments):
    """
    Save video frame masks to disk.
    
    Parameters:
    - video_dir: str, path to the directory containing video frames
    - video_segments: dict, mapping from frame_idx to {obj_id: mask}
    
    Returns: dict mapping frame_idx to list of saved mask paths
    """
    video_path = Path(video_dir)
    masks_dir = video_path / "masks"
    
    # Remove and recreate the masks directory to overwrite on each iteration
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(exist_ok=True)
    
    saved_paths = {}
    
    # Get list of frame files in order
    frame_files = sorted([f for f in video_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    for frame_idx, obj_masks in video_segments.items():
        if frame_idx >= len(frame_files):
            continue
            
        # Load the original frame
        frame_path = frame_files[frame_idx]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            continue
        
        # Create a copy for visualization
        output_frame = frame.copy()
        
        # Apply all object masks to this frame
        for obj_id, mask in obj_masks.items():
            output_frame = show_mask(output_frame, mask, random_color=True, borders=True)
        
        # Save the frame with masks
        output_filename = f"frame_{frame_idx:05d}_masks.png"
        output_path = masks_dir / output_filename
        cv2.imwrite(str(output_path), output_frame)
        
        if frame_idx not in saved_paths:
            saved_paths[frame_idx] = []
        saved_paths[frame_idx].append(str(output_path))
    
    return saved_paths


def extract_video_to_frames(video_path: Path, output_root: Path, image_extensions: set[str] | None = None) -> Path:
    """
    Extract a video into a cached frame directory.

    Parameters:
    - video_path: path to the source video file
    - output_root: root directory where extracted frame folders are stored
    - image_extensions: frame extensions considered valid for cache checks

    Returns: Path to directory containing extracted frame images
    """
    valid_image_extensions = image_extensions or {".jpg", ".jpeg", ".png", ".bmp"}

    file_stats = video_path.stat()
    cache_key = hashlib.sha1(
        f"{video_path.resolve()}:{file_stats.st_size}:{file_stats.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:12]

    output_dir = output_root / f"{video_path.stem}_{cache_key}"
    output_root.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        cached_frames = [
            frame_path
            for frame_path in output_dir.iterdir()
            if frame_path.is_file() and frame_path.suffix.lower() in valid_image_extensions
        ]
        if cached_frames:
            return output_dir
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            output_path = output_dir / f"{frame_idx:05d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write extracted frame: {output_path}")
            frame_idx += 1
    finally:
        capture.release()

    if frame_idx == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise ValueError(f"No frames could be extracted from video: {video_path}")

    return output_dir
