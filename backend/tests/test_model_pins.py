import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import co_tracker
import sam2_video_masker
from core import config

COMMIT = re.compile(r"[0-9a-f]{40}")


class ModelPinTests(unittest.TestCase):
    def test_both_models_are_pinned_to_full_commit_hashes(self):
        self.assertRegex(config.SAM2_MODEL_REVISION, COMMIT)
        self.assertRegex(config.COTRACKER_REVISION, COMMIT)

    def test_sam2_is_fetched_at_the_pinned_revision(self):
        predictor = object()
        with patch("huggingface_hub.hf_hub_download", return_value="checkpoint.pt") as download, \
                patch("sam2.build_sam.build_sam2_video_predictor", return_value=predictor) as build:
            self.assertIs(sam2_video_masker.load_predictor("cpu"), predictor)
        self.assertEqual(download.call_args.kwargs["repo_id"], config.SAM2_MODEL_ID)
        self.assertEqual(download.call_args.kwargs["revision"], config.SAM2_MODEL_REVISION)
        self.assertEqual(build.call_args.kwargs["ckpt_path"], "checkpoint.pt")
        self.assertEqual(build.call_args.kwargs["device"], "cpu")

    def test_cotracker_code_is_fetched_at_the_pinned_commit(self):
        with patch("co_tracker.torch.hub.load", return_value=MagicMock()) as load:
            co_tracker.CoTracker()
        self.assertEqual(load.call_args.args[0], f"{config.COTRACKER_REPO}:{config.COTRACKER_REVISION}")


if __name__ == "__main__":
    unittest.main()
