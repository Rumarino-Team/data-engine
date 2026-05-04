import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.propagation import propagation_batch_progress


class PropagationProgressTests(unittest.TestCase):
    def test_full_batch_reaches_batch_total(self):
        progress = propagation_batch_progress(
            processed_before_frame=31,
            window_total_frames=64,
            batch_size=32,
        )

        self.assertEqual(progress["batch_current"], 32)
        self.assertEqual(progress["batch_total"], 32)
        self.assertEqual(progress["batch_index"], 1)
        self.assertEqual(progress["batch_count"], 2)

    def test_second_batch_resets_current_count(self):
        progress = propagation_batch_progress(
            processed_before_frame=32,
            window_total_frames=64,
            batch_size=32,
        )

        self.assertEqual(progress["batch_current"], 1)
        self.assertEqual(progress["batch_total"], 32)
        self.assertEqual(progress["batch_index"], 2)
        self.assertEqual(progress["batch_count"], 2)

    def test_final_partial_batch_uses_partial_total(self):
        progress = propagation_batch_progress(
            processed_before_frame=38,
            window_total_frames=39,
            batch_size=32,
        )

        self.assertEqual(progress["batch_current"], 7)
        self.assertEqual(progress["batch_total"], 7)
        self.assertEqual(progress["batch_index"], 2)
        self.assertEqual(progress["batch_count"], 2)


if __name__ == "__main__":
    unittest.main()
