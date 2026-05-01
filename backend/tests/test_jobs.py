import sys
import threading
import time
import unittest
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.jobs as jobs


class JobTests(unittest.TestCase):
    def tearDown(self):
        with jobs.current_job_lock:
            jobs.current_job = None

    def test_queue_long_job_stores_global_job_and_completes(self):
        started = jobs.queue_long_job(
            operation="video_init",
            stage="queued",
            stage_label="Queued",
            message="queued",
            worker=lambda: {"ok": True},
        )

        deadline = time.time() + 2
        serialized = None
        while time.time() < deadline:
            serialized = jobs.serialize_job()
            if serialized and serialized.get("status") == "completed":
                break
            time.sleep(0.01)

        self.assertIsNotNone(serialized)
        self.assertEqual(serialized["job_id"], started["job_id"])
        self.assertEqual(serialized["status"], "completed")
        self.assertEqual(serialized["result"], {"ok": True})
        self.assertIsNone(serialized["batch_current"])
        self.assertIsNone(serialized["batch_total"])
        self.assertIsNone(serialized["batch_index"])
        self.assertIsNone(serialized["batch_count"])

    def test_starting_second_active_job_raises_409(self):
        release = threading.Event()
        jobs.queue_long_job(
            operation="video_init",
            stage="queued",
            stage_label="Queued",
            message="queued",
            worker=lambda: (release.wait(1), {"ok": True})[1],
        )

        deadline = time.time() + 2
        while time.time() < deadline and not jobs.is_job_active():
            time.sleep(0.01)

        with self.assertRaises(HTTPException) as context:
            jobs.queue_long_job(
                operation="mask_propagation",
                stage="queued",
                stage_label="Queued",
                message="queued",
                worker=lambda: {"ok": True},
            )
        release.set()
        self.assertEqual(context.exception.status_code, 409)

    def test_update_job_mutates_registered_job(self):
        release = threading.Event()
        started = jobs.queue_long_job(
            operation="video_init",
            stage="queued",
            stage_label="Queued",
            message="queued",
            worker=lambda: (release.wait(1), {"ok": True})[1],
        )
        jobs.update_job(
            stage="working",
            stage_label="Working",
            progress=0.25,
            batch_current=8,
            batch_total=32,
            batch_index=1,
            batch_count=4,
            message="working",
        )
        serialized = jobs.serialize_job()
        release.set()

        self.assertEqual(serialized["job_id"], started["job_id"])
        self.assertEqual(serialized["stage"], "working")
        self.assertEqual(serialized["progress"], 0.25)
        self.assertEqual(serialized["batch_current"], 8)
        self.assertEqual(serialized["batch_total"], 32)
        self.assertEqual(serialized["batch_index"], 1)
        self.assertEqual(serialized["batch_count"], 4)

    def test_failed_worker_sets_failed_status(self):
        def fail():
            raise RuntimeError("boom")

        jobs.queue_long_job(
            operation="video_init",
            stage="queued",
            stage_label="Queued",
            message="queued",
            worker=fail,
        )

        deadline = time.time() + 2
        serialized = None
        while time.time() < deadline:
            serialized = jobs.serialize_job()
            if serialized and serialized.get("status") == "failed":
                break
            time.sleep(0.01)

        self.assertEqual(serialized["status"], "failed")
        self.assertEqual(serialized["error"]["code"], "backend_error")

    def test_clear_current_job_result_slims_completed_result(self):
        started = jobs.queue_long_job(
            operation="prompt_tracking",
            stage="queued",
            stage_label="Queued",
            message="queued",
            worker=lambda: {
                "message": "done",
                "tracking_result_id": "abc",
                "tracks": [[[1, 2]]],
                "visibility": [[True]],
            },
        )

        deadline = time.time() + 2
        while time.time() < deadline:
            serialized = jobs.serialize_job()
            if serialized and serialized.get("status") == "completed":
                break
            time.sleep(0.01)

        self.assertTrue(jobs.clear_current_job_result(started["job_id"]))
        self.assertEqual(jobs.serialize_job()["result"], {"message": "done", "tracking_result_id": "abc"})


if __name__ == "__main__":
    unittest.main()
