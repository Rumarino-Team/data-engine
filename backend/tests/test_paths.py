import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import OperationError
from sessions.paths import resolve_input_path


class ResolveInputPathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="paths_")).resolve()
        (self.root / "frames").mkdir()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    def test_the_home_shortcut_is_expanded(self):
        with patch.dict(os.environ, {"HOME": str(self.root), "USERPROFILE": str(self.root)}):
            self.assertEqual(resolve_input_path("~/frames"), self.root / "frames")

    def test_environment_variables_in_a_path_are_not_expanded(self):
        with patch.dict(os.environ, {"DATA_ENGINE_TEST_ROOT": str(self.root)}):
            for written in ("%DATA_ENGINE_TEST_ROOT%/frames", "$DATA_ENGINE_TEST_ROOT/frames",
                            "${DATA_ENGINE_TEST_ROOT}/frames"):
                with self.subTest(written):
                    with self.assertRaises(OperationError) as failure:
                        resolve_input_path(written)
                    self.assertEqual(failure.exception.status_code, 404)

    def test_an_absolute_path_and_a_file_url_resolve(self):
        self.assertEqual(resolve_input_path(str(self.root / "frames")), self.root / "frames")
        self.assertEqual(resolve_input_path((self.root / "frames").as_uri()), self.root / "frames")


if __name__ == "__main__":
    unittest.main()
