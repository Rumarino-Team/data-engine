import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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


if __name__ == "__main__":
    unittest.main()
