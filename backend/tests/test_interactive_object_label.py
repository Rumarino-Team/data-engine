import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.video import VideoObject
from core.annotations import SessionAnnotations


class InteractiveObjectLabelTests(unittest.TestCase):
    def test_label_is_stripped(self):
        obj = VideoObject.model_validate({"id": 1, "label": "  buoy  ", "color": "#ff0000"})
        self.assertEqual(obj.model_dump(), {"id": 1, "label": "buoy", "color": "#ff0000"})

    def test_blank_label_is_rejected(self):
        with self.assertRaises(ValidationError):
            VideoObject.model_validate({"id": 1, "label": "   ", "color": "#ff0000"})

    def test_state_dump_uses_label(self):
        state = SessionAnnotations.model_validate(
            {"objects": {1: {"id": 1, "label": "buoy", "color": "#ff0000"}}}
        )
        self.assertNotIn("name", state.model_dump()["objects"][1])


if __name__ == "__main__":
    unittest.main()
