import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mask_manifest
from mask_manifest import frame_record, manifest_index, read_frame, read_records, write_manifest
from utils import build_empty_mask_manifest, load_mask_manifest, write_mask_manifest
from video.masks import manifest_frame_payload


def frame(seed, overlay=None, revisions=None):
    mask = np.zeros((6, 7), dtype=bool)
    mask[seed % 6, : seed % 7 + 1] = True
    payload = {**manifest_frame_payload({1: mask, 2: ~mask}), "input_revisions": revisions or {"1": seed, "2": 0}}
    if overlay:
        payload["overlay_path"] = overlay
    return payload


def header():
    manifest = build_empty_mask_manifest(source_video_path="a \"quoted\" path.mp4", resolved_video_frames_dir="/frames",
                                         num_frames=12345, frame_height=6, frame_width=7)
    manifest["session_id"] = "session"
    manifest["input_revision"] = 3
    return manifest


class ManifestFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "manifest.json"
        self.frames = {str(i): frame(i, f"overlays/frame_{i}.png" if i % 2 else None) for i in (0, 1, 2, 10, 11, 5)}
        mask_manifest._latest = None
        self.addCleanup(setattr, mask_manifest, "_latest", None)

    def write_old_way(self, path):
        write_mask_manifest(path, {**header(), "frames": self.frames})

    def write_new_way(self, path):
        write_manifest(path, header(), {key: frame_record(value) for key, value in self.frames.items()})

    def test_a_written_manifest_is_byte_for_byte_what_json_dump_wrote_and_loads_back(self):
        old, new = self.path.with_name("old.json"), self.path.with_name("new.json")
        self.write_old_way(old)
        self.write_new_way(new)
        self.assertEqual(new.read_bytes(), old.read_bytes())
        self.assertEqual(load_mask_manifest(new), {**header(), "frames": self.frames})
        self.assertEqual(list(load_mask_manifest(new)["frames"]), list(self.frames))  # frame order is kept

    def test_an_empty_manifest_round_trips(self):
        self.frames = {}
        self.write_new_way(self.path)
        self.assertEqual(load_mask_manifest(self.path), {**header(), "frames": {}})
        self.assertEqual(manifest_index(self.path), {})

    def test_the_index_of_a_written_file_is_what_scanning_it_finds(self):
        self.write_new_way(self.path)
        written = manifest_index(self.path)
        mask_manifest._latest = None
        with patch.object(mask_manifest, "_scan", wraps=mask_manifest._scan) as scan:
            scanned = manifest_index(self.path)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(written, scanned)
        self.assertEqual(list(scanned), list(self.frames))
        self.assertEqual(scanned["1"].stamps, {"1": 1, "2": 0})
        self.assertEqual(scanned["1"].overlay, "overlays/frame_1.png")
        self.assertIsNone(scanned["2"].overlay)

    def test_a_just_written_manifest_needs_no_scan(self):
        self.write_new_way(self.path)
        with patch.object(mask_manifest, "_scan", side_effect=AssertionError("scanned")):
            self.assertEqual(read_frame(self.path, "10"), self.frames["10"])
            self.assertEqual(set(manifest_index(self.path)), set(self.frames))

    def test_each_frame_is_read_alone(self):
        self.write_old_way(self.path)
        with patch.object(mask_manifest.json, "loads", wraps=json.loads) as loads:
            for key, expected in self.frames.items():
                self.assertEqual(read_frame(self.path, key), expected)
        self.assertEqual([len(call.args[0]) for call in loads.call_args_list],
                         [len(frame_record(self.frames[key]).raw) for key in self.frames])
        self.assertEqual(read_frame(self.path, "99"), {})

    def test_scanning_survives_chunk_boundaries_pretty_printing_and_frames_larger_than_a_chunk(self):
        pretty = {**header(), "frames": self.frames, "after": {"frames": 1}}
        self.path.write_text(json.dumps(pretty, indent=2), encoding="utf-8")
        reference = None
        for chunk in (1, 2, 3, 7, 64, 1 << 20):
            with self.subTest(chunk=chunk), patch.object(mask_manifest, "_CHUNK", chunk):
                mask_manifest._latest = None
                index = manifest_index(self.path)
                self.assertEqual(list(index), list(self.frames))
                for key, entry in index.items():
                    with self.path.open("rb") as handle:
                        handle.seek(entry.offset)
                        self.assertEqual(json.loads(handle.read(entry.length)), self.frames[key])
                reference = reference or index
                self.assertEqual(index, reference)

    def test_frames_may_come_before_other_top_level_values(self):
        self.path.write_text(json.dumps({"frames": self.frames, "num_frames": 7}), encoding="utf-8")
        with patch.object(mask_manifest, "_CHUNK", 5):
            self.assertEqual(list(manifest_index(self.path)), list(self.frames))
        self.path.write_text('{"version":1}')
        self.assertEqual(manifest_index(self.path), {})

    def test_a_frame_without_revisions_is_stamped_zero_and_empty_results_have_no_stamps(self):
        self.frames = {"0": {"objects": {"3": {"size": [1, 1], "rle": [], "bbox": [0, 0, 0, 0]}}}, "1": {"objects": {}}}
        self.write_old_way(self.path)
        index = manifest_index(self.path)
        self.assertEqual(index["0"].stamps, {"3": 0})
        self.assertEqual(index["1"].stamps, {})

    def test_records_of_a_file_are_its_frames_as_written(self):
        self.write_new_way(self.path)
        mask_manifest._latest = None
        records = read_records(self.path)
        self.assertEqual(list(records), list(self.frames))
        for key, record in records.items():
            self.assertEqual(record, frame_record(self.frames[key]))
        # Writing them again, with one frame replaced, gives the same file as dumping the whole manifest.
        again = self.path.with_name("again.json")
        records["1"] = frame_record(frame(1, "overlays/new.png", {"1": 9}))
        write_manifest(again, header(), records)
        self.frames["1"] = frame(1, "overlays/new.png", {"1": 9})
        expected = self.path.with_name("expected.json")
        self.write_old_way(expected)
        self.assertEqual(again.read_bytes(), expected.read_bytes())

    def test_invalid_and_truncated_files_raise_instead_of_reading_as_empty(self):
        for text in ("invalid JSON", "", "[]", '{"frames":[]}', '{"frames":{"0":{"objects":', '{"frames":{"0":{}'):
            with self.subTest(text=text):
                self.path.write_text(text)
                mask_manifest._latest = None
                with self.assertRaises(json.JSONDecodeError):
                    manifest_index(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            manifest_index(self.path)


class ManifestIndexCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "manifest.json"
        mask_manifest._latest = None
        self.addCleanup(setattr, mask_manifest, "_latest", None)

    def put(self, name, text):
        path = self.path.with_name(name)
        path.write_text(text)
        return path

    def test_reuses_the_index_of_an_unchanged_file(self):
        self.path.write_text('{"frames":{"0":{"objects":{}}}}')
        with patch.object(mask_manifest, "_scan", wraps=mask_manifest._scan) as scan:
            first = manifest_index(self.path)
            second = manifest_index(self.path)
        self.assertIs(first, second)
        self.assertEqual(scan.call_count, 1)

    def test_rescans_after_same_size_in_place_rewrite_with_new_mtime(self):
        self.path.write_text('{"frames":{"0":{"objects":{"1":{}}}}}')
        original = self.path.stat()
        self.assertEqual(manifest_index(self.path)["0"].stamps, {"1": 0})
        self.path.write_text('{"frames":{"7":{"objects":{"1":{}}}}}')
        # Coarse filesystem timestamps can tie, so make the modification time distinct.
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        self.assertEqual(list(manifest_index(self.path)), ["7"])

    def test_rescans_same_size_atomic_replacement_even_with_restored_mtime(self):
        self.path.write_text('{"frames":{"0":{"objects":{}}}}')
        original = self.path.stat()
        self.assertEqual(list(manifest_index(self.path)), ["0"])
        replacement = self.put("replacement.json", '{"frames":{"9":{"objects":{}}}}')
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(self.path)
        self.assertEqual(list(manifest_index(self.path)), ["9"])
        self.assertEqual(read_frame(self.path, "9"), {"objects": {}})

    def test_keeps_only_the_latest_manifest(self):
        self.path.write_text('{"frames":{"0":{}}}')
        manifest_index(self.path)
        other = self.put("other.json", '{"frames":{"1":{}}}')
        manifest_index(other)
        with patch.object(mask_manifest, "_scan", wraps=mask_manifest._scan) as scan:
            manifest_index(self.path)
        self.assertEqual(scan.call_count, 1)


if __name__ == "__main__":
    unittest.main()
