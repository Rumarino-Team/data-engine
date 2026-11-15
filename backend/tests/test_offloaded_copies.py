"""SAM2 copies results from the GPU to CPU memory without waiting, then reads them at once.

With the video state offloaded (the default), that race made a second object's first click return an empty mask or
the first object's mask from another frame. The predictor's methods that start those copies are wrapped to wait for
them. The unit tests cover the wrapper; the last test runs the real model on the sample clip and is skipped when
CUDA, the pinned weights or the frames are missing."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sam2_video_masker
from core import config
from sam2_video_masker import SAM2VideoMasker, make_offloaded_copies_safe, mask_logits_to_2d_bool

BEDROOM = Path(__file__).resolve().parent / "bedroom"


def state(device, storage):
    return {"device": torch.device(device), "storage_device": torch.device(storage)}


class Stream:
    def __init__(self, events):
        self.events = events

    def synchronize(self):
        self.events.append("wait")


class WaitingForCopiesTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        patcher = patch("sam2_video_masker.torch.cuda.current_stream", side_effect=lambda device: Stream(self.events))
        patcher.start()
        self.addCleanup(patcher.stop)

        def inference(inference_state, frame_idx):
            self.events.append("inference")
            return "mask"

        def encoder(inference_state, frame_idx):
            self.events.append("encoder")
            return "features"

        self.predictor = make_offloaded_copies_safe(
            SimpleNamespace(_run_single_frame_inference=inference, _run_memory_encoder=encoder))

    def test_it_waits_after_the_copies_start_and_passes_the_result_through(self):
        offloaded = state("cuda", "cpu")
        self.assertEqual(self.predictor._run_single_frame_inference(inference_state=offloaded, frame_idx=3), "mask")
        self.assertEqual(self.predictor._run_memory_encoder(offloaded, 3), "features")  # state passed positionally
        self.assertEqual(self.events, ["inference", "wait", "encoder", "wait"])

    def test_it_does_not_wait_when_nothing_is_copied_to_the_cpu(self):
        for name, device, storage in (("state kept on the GPU", "cuda", "cuda"), ("running on the CPU", "cpu", "cpu")):
            with self.subTest(name):
                self.events.clear()
                self.predictor._run_single_frame_inference(inference_state=state(device, storage), frame_idx=0)
                self.assertEqual(self.events, ["inference"])

    def test_a_predictor_missing_one_of_the_methods_is_refused(self):
        with self.assertRaises(AttributeError):
            make_offloaded_copies_safe(SimpleNamespace(_run_single_frame_inference=lambda **kwargs: None))


def weights_are_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
        _, checkpoint = HF_MODEL_ID_TO_FILENAMES[config.SAM2_MODEL_ID]
        found = try_to_load_from_cache(config.SAM2_MODEL_ID, checkpoint, revision=config.SAM2_MODEL_REVISION)
        return isinstance(found, str)
    except Exception:
        return False


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless((BEDROOM / "00006.jpg").is_file(), "requires the sample frames in tests/bedroom")
@unittest.skipUnless(weights_are_cached(), "requires the pinned SAM2 weights in the Hugging Face cache")
class SecondObjectClickTests(unittest.TestCase):
    BOY, GIRL = [[225, 275]], [[395, 235]]

    def click(self, masker, frame, obj_id, points):
        return masker.add_new_points_or_box(
            frame_idx=frame, obj_id=obj_id, points=points, labels=[1] * len(points), clear_old_points=True, box=None)

    def test_a_second_objects_first_click_on_another_frame_segments_what_was_clicked(self):
        masker = SAM2VideoMasker()
        for attempt in range(5):  # the race was intermittent, so one lucky run proves nothing
            masker.init_state(str(BEDROOM), online_mode=True, offload_state_to_cpu=True)
            self.click(masker, 0, 1, self.BOY)
            _, obj_ids, logits = self.click(masker, 6, 2, self.GIRL)
            mask = mask_logits_to_2d_bool(logits[list(map(int, obj_ids)).index(2)])
            with self.subTest(attempt=attempt):
                self.assertTrue(mask[235, 395], f"the mask ({int(mask.sum())} px) does not contain the click")


if __name__ == "__main__":
    unittest.main()
