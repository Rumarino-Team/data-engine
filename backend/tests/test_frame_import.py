"""Loading a folder of frames: JPEG files are used as they are, PNG and BMP are converted, because SAM2 reads only JPEG."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import OperationError
from core.jobs import JobHandle
from sam2_frame_store import list_frame_paths
from video import loading
from video.io import copy_frames_directory_to_session

SHAPE = (8, 12, 3)


class FrameImportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="frame_import_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "source"
        self.frames = self.root / "frames"
        self.source.mkdir()
        self.frames.mkdir()

    def write(self, name, color, shape=SHAPE):
        image = np.full(shape, color, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.source / name), image))

    def decoded(self, name):
        image = cv2.imread(str(self.frames / name))
        self.assertIsNotNone(image, name)
        return image

    def test_png_and_bmp_become_numbered_jpegs_next_to_jpegs_that_are_kept(self):
        self.write("a.png", (10, 20, 30))
        self.write("b.bmp", (40, 50, 60))
        self.write("c.jpg", (70, 80, 90))
        names = copy_frames_directory_to_session(self.source, self.frames)
        self.assertEqual(names, ["00000.jpg", "00001.jpg", "00002.jpg"])
        for name, color in zip(names, [(10, 20, 30), (40, 50, 60), (70, 80, 90)]):
            image = self.decoded(name)
            self.assertEqual(image.shape, SHAPE)
            self.assertLessEqual(int(np.abs(image.astype(int) - np.array(color)).max()), 3)
        # A JPEG is not decoded and written again, so it stays byte for byte what the user gave.
        self.assertEqual((self.frames / "00002.jpg").read_bytes(), (self.source / "c.jpg").read_bytes())

    def test_a_folder_of_only_jpegs_keeps_its_file_types(self):
        self.write("a.jpg", (1, 2, 3))
        self.write("b.jpeg", (4, 5, 6))
        self.assertEqual(copy_frames_directory_to_session(self.source, self.frames), ["00000.jpg", "00001.jpeg"])

    def test_alpha_and_grayscale_images_become_three_channel_frames(self):
        self.write("a.png", (10, 20, 30, 128), shape=(8, 12, 4))
        self.write("b.png", 77, shape=(8, 12))
        names = copy_frames_directory_to_session(self.source, self.frames)
        self.assertEqual([self.decoded(name).shape for name in names], [SHAPE, SHAPE])

    def test_converted_frames_are_what_sam2_s_frame_store_can_list(self):
        self.write("a.png", (10, 20, 30))
        self.write("b.bmp", (40, 50, 60))
        copy_frames_directory_to_session(self.source, self.frames)
        self.assertEqual([Path(path).name for path in list_frame_paths(str(self.frames))], ["00000.jpg", "00001.jpg"])

    def test_an_unreadable_image_is_a_400_that_names_the_file(self):
        self.write("a.png", (10, 20, 30))
        (self.source / "b.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, r"b\.png"):
            copy_frames_directory_to_session(self.source, self.frames)
        with self.assertRaises(OperationError) as failure:
            loading._link_frame_directory(self.source, self.frames, Mock(spec=JobHandle))
        self.assertEqual(failure.exception.status_code, 400)
        self.assertIn("b.png", failure.exception.message)

    def test_the_job_says_when_frames_are_converted(self):
        self.write("a.png", (10, 20, 30))
        self.write("b.jpg", (40, 50, 60))
        job = Mock(spec=JobHandle)
        loading._link_frame_directory(self.source, self.frames, job)
        first = job.enter_stage.call_args_list[0].kwargs["message"]
        self.assertIn("1 of 2 are PNG/BMP and will be converted to JPEG", first)


if __name__ == "__main__":
    unittest.main()
