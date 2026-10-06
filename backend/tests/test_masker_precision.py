import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2_video_masker import SAM2VideoMasker


class PrecisionPredictor:
    """Cache reduced-precision features, then multiply by FP32 weights later."""

    def __init__(self, device):
        self.device = device
        self.weight = torch.ones((4, 4), device=device)
        self.calls = []
        self.fail = False

    def compute(self, name, features=None):
        self.calls.append((name, torch.is_autocast_enabled(self.device),
                           torch.is_inference_mode_enabled()))
        if self.fail:
            raise RuntimeError("predictor failed")
        return (self.weight if features is None else features) @ self.weight

    def init_state(self, **kwargs):
        return {"num_frames": 3, "features": self.compute("init")}

    def reset_state(self, state):
        self.compute("reset", state["features"])

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
        return frame_idx, [obj_id], self.compute("points", inference_state["features"])

    def add_new_mask(self, inference_state, frame_idx, obj_id, **kwargs):
        return frame_idx, [obj_id], self.compute("mask", inference_state["features"])

    def propagate_in_video_preflight(self, state):
        self.compute("preflight", state["features"])

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse, **kwargs):
        for i in range(start_frame_idx, min(3, start_frame_idx + max_frame_num_to_track + 1)):
            # This executes when the generator advances, not when it is created.
            yield i, [1], self.compute("propagate", state["features"])[None, None]

    def clear_all_prompts_in_frame(self, inference_state, **kwargs):
        self.compute("clear", inference_state["features"])

    def remove_object(self, inference_state, **kwargs):
        self.compute("remove", inference_state["features"])


def fresh_thread(operation):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(operation).result()


class MaskerPrecisionTests(unittest.TestCase):
    def make_masker(self, device):
        predictor = PrecisionPredictor(device)
        with patch("sam2_video_masker.torch.cuda.is_available", return_value=device == "cuda"), \
                patch("sam2_video_masker.SAM2VideoPredictor.from_pretrained", return_value=predictor):
            masker = SAM2VideoMasker()
        with tempfile.TemporaryDirectory() as directory:
            masker.init_state(directory)
        self.assertFalse(torch.is_autocast_enabled(device))
        self.assertFalse(torch.is_inference_mode_enabled())
        return masker

    def test_cpu_uses_full_precision_without_enabling_autocast(self):
        masker = fresh_thread(lambda: self.make_masker("cpu"))
        _, _, result = masker.add_new_points_or_box(0, 1)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(all(not autocast and inference for _, autocast, inference in masker.predictor.calls))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_context_survives_switch_from_init_thread_to_request_thread(self):
        masker = fresh_thread(lambda: self.make_masker("cuda"))
        features = masker.inference_state["features"]
        self.assertEqual(features.dtype, masker.autocast_dtype)
        # Demonstrate the original failure without autocast using these exact tensors.
        with self.assertRaises(RuntimeError):
            features @ masker.predictor.weight
        _, _, result = masker.add_new_points_or_box(0, 1)
        self.assertEqual(result.dtype, masker.autocast_dtype)
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_both_propagation_paths_cover_lazy_iteration_in_a_new_job_thread(self):
        for online in [False, True]:
            with self.subTest(online=online):
                masker = fresh_thread(lambda: self.make_masker("cuda"))
                def propagate():
                    result = masker.propagate_in_video(
                        start_frame_idx=0, max_frame_num_to_track=3,
                        online_mode=online, batch_size=1,
                    )
                    self.assertFalse(torch.is_autocast_enabled("cuda"))
                    return result
                result = fresh_thread(propagate)
                self.assertEqual(set(result), {0, 1, 2})
                self.assertTrue(all(autocast and inference for _, autocast, inference in masker.predictor.calls))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_other_predictor_entry_points_use_the_same_precision(self):
        masker = fresh_thread(lambda: self.make_masker("cuda"))
        masker.add_new_mask(0, 1, [[True]])
        masker.clear_all_prompts_in_frame(0, 1)
        masker.remove_object(1)
        masker.reset_state()
        self.assertTrue(all(autocast and inference for _, autocast, inference in masker.predictor.calls))
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_without_bfloat16_support_uses_scoped_float16(self):
        with patch("sam2_video_masker.torch.cuda.is_bf16_supported", return_value=False):
            masker = fresh_thread(lambda: self.make_masker("cuda"))
        _, _, result = masker.add_new_points_or_box(0, 1)
        self.assertEqual(masker.autocast_dtype, torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_exception_restores_the_callers_context(self):
        masker = fresh_thread(lambda: self.make_masker("cuda"))
        masker.predictor.fail = True
        with torch.autocast("cuda", dtype=torch.float16):
            with self.assertRaisesRegex(RuntimeError, "predictor failed"):
                masker.propagate_in_video(start_frame_idx=0, max_frame_num_to_track=1, online_mode=False)
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            self.assertEqual(torch.get_autocast_dtype("cuda"), torch.float16)
            self.assertFalse(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_autocast_enabled("cuda"))


if __name__ == "__main__":
    unittest.main()
