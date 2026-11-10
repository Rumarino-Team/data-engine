import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import OperationError
from core.jobs import JobManager


class JobManagerTests(unittest.TestCase):
    """Each test owns its manager: nothing here touches the application's."""

    def setUp(self):
        self.jobs = JobManager(threading.RLock())
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def start(self, worker, operation="video_init"):
        return self.jobs.start(
            operation=operation, stage="queued", stage_label="Queued", message="queued", worker=worker)

    def wait_for(self, status):
        deadline = time.time() + 2
        while time.time() < deadline:
            job = self.jobs.snapshot()
            if job and job["status"] == status:
                return job
            time.sleep(0.005)
        self.fail(f"job never reached {status}: {self.jobs.snapshot()}")

    def hold(self, operation="video_init"):
        """Start a job that stays running until the test ends; returns its handle."""
        handles = []
        started = threading.Event()

        def worker(handle):
            handles.append(handle)
            started.set()
            self.release.wait(2)
            return {"ok": True}
        self.start(worker, operation)
        self.assertTrue(started.wait(2))
        return handles[0]

    def test_job_runs_on_a_worker_and_completes_with_its_result(self):
        started = self.start(lambda handle: {"ok": True})
        job = self.wait_for("completed")
        self.assertEqual((job["job_id"], job["result"]), (started["job_id"], {"ok": True}))
        for field in ("batch_current", "batch_total", "batch_index", "batch_count", "window_index"):
            self.assertIsNone(job[field], field)

    def test_a_second_active_job_is_refused_while_one_runs(self):
        self.hold()
        with self.assertRaises(OperationError) as refused:
            self.start(lambda handle: {}, "mask_propagation")
        self.assertEqual(refused.exception.status_code, 409)
        with self.assertRaises(OperationError):
            self.jobs.require_idle()
        self.release.set()
        self.wait_for("completed")
        self.jobs.require_idle()

    def test_update_refines_a_stage_and_entering_one_clears_the_previous_position(self):
        handle = self.hold()
        handle.update(stage="working", progress=0.25, batch_current=8, batch_total=32, window_index=1, frame_idx=9)
        job = self.jobs.snapshot()
        self.assertEqual((job["stage"], job["progress"], job["batch_current"], job["batch_total"]),
                         ("working", 0.25, 8, 32))

        handle.enter_stage("saving", "Saving", message="saving", progress=0.9, total=5)
        job = self.jobs.snapshot()
        self.assertEqual((job["stage"], job["progress"], job["total"]), ("saving", 0.9, 5))
        for field in ("current", "frame_idx", "window_index", "batch_current", "batch_total"):
            self.assertIsNone(job[field], field)
        with self.assertRaises(TypeError):
            handle.enter_stage("saving", "Saving", message="saving", frame=1)

    def test_a_finished_workers_handle_cannot_touch_a_newer_job(self):
        first = self.hold()
        self.release.set()
        self.wait_for("completed")
        self.release.clear()
        second = self.hold()
        first.enter_stage("stale", "Stale", message="stale")
        self.assertEqual(self.jobs.snapshot()["job_id"], second.job_id)
        self.assertEqual(self.jobs.snapshot()["stage"], "queued")

    def test_failed_worker_sets_failed_status(self):
        def fail(handle):
            raise RuntimeError("boom")
        self.start(fail)
        job = self.wait_for("failed")
        self.assertEqual(job["error"]["code"], "backend_error")

    def test_clearing_a_finished_result_keeps_only_its_summary(self):
        started = self.start(lambda handle: {
            "message": "done", "tracking_result_id": "abc", "tracks": [[[1, 2]]], "visibility": [[True]],
        }, "prompt_tracking")
        self.wait_for("completed")
        self.assertTrue(self.jobs.clear_result(started["job_id"]))
        self.assertEqual(self.jobs.snapshot()["result"], {"message": "done", "tracking_result_id": "abc"})
        self.assertFalse(self.jobs.clear_result("another"))

    def test_a_running_job_cannot_be_cleared(self):
        handle = self.hold()
        self.assertFalse(self.jobs.clear_result(handle.job_id))


if __name__ == "__main__":
    unittest.main()
