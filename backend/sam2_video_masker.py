import torch
import numpy as np
import os
import gc
from pathlib import Path
from typing import Callable, Optional
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils import extract_video_to_frames

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_FRAMES_ROOT = PROJECT_ROOT / "backend/cache/frames"


class SAM2VideoMasker:
    def __init__(self, progress_callback: Optional[Callable[[str, str, Optional[float], str], None]] = None):
        def _report(stage: str, label: str, progress: Optional[float], message: str) -> None:
            if progress_callback is not None:
                progress_callback(stage, label, progress, message)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Utilizing device: {self.device}")

        if self.device.type == "cuda":
            device_props = torch.cuda.get_device_properties(0)
            gpu_name = device_props.name
            compute_capability = (int(device_props.major), int(device_props.minor))

            # Ampere (RTX 30 series) and newer support TF32 and practical bf16 inference.
            is_ampere_or_newer = compute_capability[0] >= 8
            supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            use_bf16 = is_ampere_or_newer and supports_bf16

            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            torch.autocast("cuda", dtype=autocast_dtype).__enter__()

            torch.backends.cuda.matmul.allow_tf32 = is_ampere_or_newer
            torch.backends.cudnn.allow_tf32 = is_ampere_or_newer

            if is_ampere_or_newer:
                gpu_family = "RTX 30-series+ / Ampere+"
            elif compute_capability[0] == 7:
                gpu_family = "RTX 20-series / Turing"
            else:
                gpu_family = "pre-RTX 20-series architecture"

            print(
                "CUDA precision config | "
                f"GPU: {gpu_name} (cc {compute_capability[0]}.{compute_capability[1]}) | "
                f"family: {gpu_family} | "
                f"autocast: {autocast_dtype} | "
                f"tf32: {is_ampere_or_newer}"
            )

        _report("loading_sam2_model", "Loading SAM2 model", None, "Loading SAM2 model weights")
        self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=self.device)
        _report("model_ready", "SAM2 model ready", 0.35, "SAM2 model loaded")

        self.inference_state = None
        self.online_mode = True
        self.default_batch_size = 32
        self.offload_video_to_cpu = True
        self.offload_state_to_cpu = False
    
    def init_state(
        self,
        video_dir,
        online_mode=True,
        batch_size=None,
        offload_video_to_cpu=None,
        offload_state_to_cpu=None,
        async_loading_frames=False,
        progress_callback: Optional[Callable[[str, str, Optional[float], str], None]] = None,
    ):
        self.online_mode = bool(online_mode)

        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            self.default_batch_size = batch_size

        if offload_video_to_cpu is None:
            offload_video_to_cpu = self.online_mode
        if offload_state_to_cpu is None:
            offload_state_to_cpu = self.online_mode

        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)

        resolved_input_path = Path(video_dir).expanduser()
        if not resolved_input_path.is_absolute():
            resolved_input_path = (Path.cwd() / resolved_input_path).resolve()
        else:
            resolved_input_path = resolved_input_path.resolve()

        if not resolved_input_path.exists():
            raise ValueError(f"Path not found: {resolved_input_path}")

        if resolved_input_path.is_file():
            suffix = resolved_input_path.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                resolved_video_dir = extract_video_to_frames(
                    resolved_input_path,
                    output_root=GENERATED_FRAMES_ROOT,
                    image_extensions=IMAGE_EXTENSIONS,
                )
            elif suffix in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Expected a frames directory or video file, got image file: {resolved_input_path}."
                )
            else:
                raise ValueError(
                    f"Unsupported input file type: {resolved_input_path.suffix or '<none>'}."
                )
        else:
            resolved_video_dir = resolved_input_path

        if progress_callback is not None:
            progress_callback(
                "initializing_sam2_video_state",
                "Initializing SAM2 video state",
                None,
                "Loading frames into SAM2 state",
            )

        self.inference_state = self.predictor.init_state(
            video_path=str(resolved_video_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        self.predictor.reset_state(self.inference_state)
        if progress_callback is not None:
            progress_callback(
                "indexing_frames",
                "Indexing video frames",
                0.85,
                "SAM2 state initialized; indexing frame files",
            )

    def reset_state(self):
        self.predictor.reset_state(self.inference_state)

    def add_new_points_or_box(self, frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
        out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            box=box,
        )

        return out_frame_idx, out_obj_ids, out_mask_logits

    def add_new_mask(self, frame_idx, obj_id, mask):
        """Add new mask to a frame."""
        frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

        return frame_idx, out_obj_ids, out_mask_logits

    def _required_non_cond_history(self):
        num_maskmem = max(1, int(getattr(self.predictor, "num_maskmem", 1)))
        stride = max(1, int(getattr(self.predictor, "memory_temporal_stride_for_eval", 1)))

        if num_maskmem <= 1:
            memory_window = 0
        else:
            memory_window = max(num_maskmem - 1, (num_maskmem - 2) * stride)

        pointer_window = 0
        if bool(getattr(self.predictor, "use_obj_ptrs_in_encoder", False)):
            pointer_window = max(0, int(getattr(self.predictor, "max_obj_ptrs_in_encoder", 0)) - 1)

        return max(memory_window, pointer_window)

    def _purge_non_conditioning_outputs(self, anchor_frame_idx, reverse=False):
        if self.inference_state is None:
            return

        keep_window = self._required_non_cond_history() + 2
        lower_bound = anchor_frame_idx - keep_window
        upper_bound = anchor_frame_idx + keep_window

        def _purge_non_cond_dict(per_obj_dict):
            for obj_output_dict in per_obj_dict.values():
                non_cond_outputs = obj_output_dict.get("non_cond_frame_outputs", {})
                if reverse:
                    stale_keys = [
                        frame_idx
                        for frame_idx in list(non_cond_outputs.keys())
                        if frame_idx < anchor_frame_idx or frame_idx > upper_bound
                    ]
                else:
                    stale_keys = [
                        frame_idx
                        for frame_idx in list(non_cond_outputs.keys())
                        if frame_idx > anchor_frame_idx or frame_idx < lower_bound
                    ]
                for frame_idx in stale_keys:
                    non_cond_outputs.pop(frame_idx, None)

        _purge_non_cond_dict(self.inference_state.get("output_dict_per_obj", {}))
        _purge_non_cond_dict(self.inference_state.get("temp_output_dict_per_obj", {}))

        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _resolve_start_frame_idx(self, start_frame_idx):
        if start_frame_idx is not None:
            return int(start_frame_idx)

        return min(
            frame_idx
            for obj_output_dict in self.inference_state["output_dict_per_obj"].values()
            for frame_idx in obj_output_dict["cond_frame_outputs"]
        )

    def _compute_total_frames_to_process(
        self,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        num_frames = int(self.inference_state["num_frames"])
        if num_frames <= 0:
            return 0

        if reverse:
            clamped_start = min(max(int(start_frame_idx), 0), num_frames - 1)
            available = clamped_start + 1
        else:
            clamped_start = min(max(int(start_frame_idx), 0), num_frames)
            available = num_frames - clamped_start

        if max_frame_num_to_track is None:
            return available

        requested = int(max_frame_num_to_track)
        if requested <= 0:
            return 0

        return min(available, requested)

    def _propagate_in_video_batched(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        batch_size=None,
        collect_segments=True,
        frame_callback=None,
    ):
        if self.inference_state is None:
            return {}

        self.predictor.propagate_in_video_preflight(self.inference_state)

        effective_batch_size = self.default_batch_size if batch_size is None else int(batch_size)
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        num_frames = int(self.inference_state["num_frames"])
        current_start = self._resolve_start_frame_idx(start_frame_idx)
        total_frames_to_process = self._compute_total_frames_to_process(
            start_frame_idx=current_start,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        if total_frames_to_process <= 0:
            return {}

        total_remaining = total_frames_to_process
        video_segments = {} if collect_segments else None
        global_last_processed_frame_idx = None

        while True:
            if total_remaining <= 0:
                break

            frames_this_batch = min(effective_batch_size, total_remaining)

            # SAM2's max_frame_num_to_track is inclusive with start_frame_idx.
            # To process exactly N frames in a batch, pass N-1 here.
            predictor_max_frames = max(0, int(frames_this_batch) - 1)

            processed_in_batch = 0
            last_processed_frame_idx = None
            processed_before_batch = total_frames_to_process - total_remaining

            propagate_kwargs = {
                "start_frame_idx": current_start,
                "max_frame_num_to_track": predictor_max_frames,
                "reverse": reverse,
            }

            def _iter_with_progress_fallback():
                try:
                    yield from self.predictor.propagate_in_video(
                        self.inference_state,
                        progress_total=total_frames_to_process,
                        progress_initial=processed_before_batch,
                        **propagate_kwargs,
                    )
                    return
                except TypeError as error:
                    # Backward compatibility with SAM2 predictor builds that do
                    # not support progress_* kwargs.
                    error_message = str(error)
                    if "progress_total" not in error_message and "progress_initial" not in error_message:
                        raise

                yield from self.predictor.propagate_in_video(
                    self.inference_state,
                    **propagate_kwargs,
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in _iter_with_progress_fallback():
                if collect_segments:
                    if out_frame_idx not in video_segments:
                        processed_in_batch += 1
                else:
                    processed_in_batch += 1
                frame_masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                if collect_segments:
                    video_segments[out_frame_idx] = frame_masks
                if frame_callback is not None:
                    frame_callback(out_frame_idx, frame_masks)
                last_processed_frame_idx = out_frame_idx

            if processed_in_batch == 0 or last_processed_frame_idx is None:
                break

            if global_last_processed_frame_idx is not None:
                if reverse and last_processed_frame_idx >= global_last_processed_frame_idx:
                    break
                if not reverse and last_processed_frame_idx <= global_last_processed_frame_idx:
                    break
            global_last_processed_frame_idx = last_processed_frame_idx

            total_remaining -= processed_in_batch

            self._purge_non_conditioning_outputs(last_processed_frame_idx, reverse=reverse)

            if reverse:
                next_start = last_processed_frame_idx - 1
                if next_start < 0:
                    break
            else:
                next_start = last_processed_frame_idx + 1
                if next_start >= num_frames:
                    break

            current_start = next_start

        return video_segments if collect_segments else {}

    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        batch_size=None,
        online_mode=None,
        collect_segments=True,
        frame_callback=None,
    ):
        use_online_mode = self.online_mode if online_mode is None else bool(online_mode)
        if use_online_mode:
            return self._propagate_in_video_batched(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
                batch_size=batch_size,
                collect_segments=collect_segments,
                frame_callback=frame_callback,
            )

        video_segments = {} if collect_segments else None
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        ):
            frame_masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if collect_segments:
                video_segments[out_frame_idx] = frame_masks
            if frame_callback is not None:
                frame_callback(out_frame_idx, frame_masks)

        return video_segments if collect_segments else {}

    def clear_all_prompts_in_frame(self, frame_idx, obj_id):
        self.predictor.clear_all_prompts_in_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx, obj_id=obj_id
        )
        return

    def remove_object(self, obj_id):
        self.predictor.remove_object(
            inference_state=self.inference_state,
            obj_id=obj_id
        )
