import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import OperationError
from sessions.paths import manifest_relative_path, resolve_input_path


class ResolveInputPathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="paths_")).resolve()
        (self.root / "frames").mkdir()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    def test_the_home_shortcut_is_expanded(self):
        with patch.dict(os.environ, {"HOME": str(self.root), "USERPROFILE": str(self.root)}):
            self.assertEqual(resolve_input_path("~/frames"), self.root / "frames")

    def test_an_absolute_path_and_a_file_url_resolve(self):
        self.assertEqual(resolve_input_path(str(self.root / "frames")), self.root / "frames")
        self.assertEqual(resolve_input_path((self.root / "frames").as_uri()), self.root / "frames")


if __name__ == "__main__":
    unittest.main()
