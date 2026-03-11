import torch
import numpy as np
import os
from pathlib import Path
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils import extract_video_to_frames

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_FRAMES_ROOT = PROJECT_ROOT / "backend/.data_engine_frames"


class SAM2VideoMasker:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Utilizing device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # enable tf32 for new GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

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
            offload_state_to_cpu = False

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

        self.inference_state = self.predictor.init_state(
            video_path=str(resolved_video_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        self.predictor.reset_state(self.inference_state)

    def reset_state(self):
        self.predictor.reset_state(self.inference_state)

    def add_new_points_or_box(self, frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            box=box,
        )

        return out_obj_ids, out_mask_logits

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

        output_dict_per_obj = self.inference_state.get("output_dict_per_obj", {})
        for obj_output_dict in output_dict_per_obj.values():
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
        video_segments = {}
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

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=current_start,
                max_frame_num_to_track=predictor_max_frames,
                reverse=reverse,
                progress_total=total_frames_to_process,
                progress_initial=processed_before_batch,
            ):
                if out_frame_idx not in video_segments:
                    processed_in_batch += 1
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
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

        return video_segments

    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        batch_size=None,
        online_mode=None,
    ):
        use_online_mode = self.online_mode if online_mode is None else bool(online_mode)
        if use_online_mode:
            return self._propagate_in_video_batched(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
                batch_size=batch_size,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments

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