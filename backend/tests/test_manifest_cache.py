import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.manifest_cache import _read_manifest, load_cached_mask_manifest


class ManifestCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.json"
        _read_manifest.cache_clear()

    def tearDown(self):
        _read_manifest.cache_clear()
        self.tmp.cleanup()

    def test_reuses_parsed_manifest_for_unchanged_file(self):
        self.path.write_text('{"frames":{"0":{"objects":{}}}}')
        with patch("video.manifest_cache.json.load", wraps=json.load) as parse:
            first = load_cached_mask_manifest(self.path)
            second = load_cached_mask_manifest(self.path)
        self.assertIs(first, second)
        self.assertEqual(parse.call_count, 1)

    def test_reloads_after_same_size_rewrite_even_with_restored_mtime(self):
        self.path.write_text('{"version":1}')
        original = self.path.stat()
        self.assertEqual(load_cached_mask_manifest(self.path)["version"], 1)
        self.path.write_text('{"version":2}')
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(load_cached_mask_manifest(self.path)["version"], 2)

    def test_reloads_replaced_file_and_keeps_only_one_manifest(self):
        self.path.write_text('{"version":1}')
        load_cached_mask_manifest(self.path)
        replacement = self.path.with_name("replacement.json")
        replacement.write_text('{"version":2}')
        replacement.replace(self.path)
        self.assertEqual(load_cached_mask_manifest(self.path)["version"], 2)
        other = self.path.with_name("other.json")
        other.write_text('{"version":3}')
        load_cached_mask_manifest(other)
        self.assertEqual(_read_manifest.cache_info().currsize, 1)

    def test_missing_and_invalid_files_are_not_served_from_old_cache(self):
        self.path.write_text('{"version":1}')
        load_cached_mask_manifest(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_cached_mask_manifest(self.path)
        self.path.write_text("invalid JSON")
        with self.assertRaises(json.JSONDecodeError):
            load_cached_mask_manifest(self.path)


if __name__ == "__main__":
    unittest.main()
