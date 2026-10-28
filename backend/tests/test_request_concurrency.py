import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import anyio.to_thread
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.factory import create_app
from core import jobs
from core.state import BackendState, state
from routers import video as video_routes
from sessions.metadata import write_session_metadata
from video import service


class RequestConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """All requests share one ASGI loop, unlike separate TestClient portals."""

    async def asyncSetUp(self):
        self.previous_state = state.__dict__.copy()
        with jobs.current_job_lock:
            self.previous_job = jobs.current_job
            jobs.current_job = None
        self.tmp = tempfile.TemporaryDirectory()
        self.session_path = Path(self.tmp.name)
        state.__dict__.update(BackendState(
            active_session_id="session", active_session_dir=self.session_path,
            video_dir=str(self.session_path), video_frame_files=["00000.jpg"],
            video_masker=Mock(online_mode=True, default_batch_size=1,
                              offload_video_to_cpu=True, offload_state_to_cpu=True),
        ).__dict__)
        state.annotations.create_object()
        write_session_metadata()
        job = jobs._start_job("video_init", stage="init", stage_label="Init", message="init")
        jobs._complete_job(job["job_id"], {"message": "previous job"})
        self.job_id = job["job_id"]
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test",
        )
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.worker_threads = []
        self.requests = []

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*self.requests, return_exceptions=True), timeout=5)
        await self.client.aclose()
        state.__dict__.clear()
        state.__dict__.update(self.previous_state)
        with jobs.current_job_lock:
            jobs.current_job = self.previous_job
        self.tmp.cleanup()

    def block_worker(self):
        self.worker_threads.append(threading.get_ident())
        self.loop.call_soon_threadsafe(self.entered.set)
        # A watchdog keeps a regression from hanging the suite if the ASGI loop
        # accidentally executes this blocking operation itself.
        if not self.release.wait(timeout=5):
            raise TimeoutError("The blocked worker was never released.")

    def start_request(self, method, path, **kwargs):
        task = asyncio.create_task(self.client.request(method, path, **kwargs))
        self.requests.append(task)
        return task

    def start_create(self):
        return self.start_request("POST", "/video/objects", headers={
            "X-Session-ID": "session", "X-Annotation-Revision": "1",
        })

    async def assert_control_requests_respond(self):
        paths = ["/health", "/jobs/current", f"/jobs/{self.job_id}"]
        responses = await asyncio.wait_for(
            asyncio.gather(*(self.client.get(path) for path in paths)), timeout=1,
        )
        self.assertEqual([response.status_code for response in responses], [200, 200, 200])
        self.assertEqual(responses[2].json()["job"]["status"], "completed")
        self.assertFalse(self.release.is_set())

    async def test_health_and_polling_respond_while_interactive_inference_is_blocked(self):
        def infer(**kwargs):
            self.block_worker()
            return 0, [1], [np.ones((2, 2), dtype=bool)]
        state.video_masker.add_new_points_or_box.side_effect = infer
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask):
            edit = self.start_request("POST", "/video/add_new_points_or_box", headers={
                "X-Session-ID": "session", "X-Annotation-Revision": "1",
            }, json={"frame_idx": 0, "obj_id": 1, "points": [[1, 1]], "labels": [1]})
            try:
                await asyncio.wait_for(self.entered.wait(), timeout=1)
                self.assertNotEqual(self.worker_threads[0], self.loop_thread)
                await self.assert_control_requests_respond()
                self.assertFalse(edit.done())
            finally:
                self.release.set()
                response = await asyncio.wait_for(edit, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["annotation_delta"]["revision"], 2)

    async def test_concurrent_mutation_and_snapshot_wait_for_metadata_acceptance(self):
        second_entered, read_entered = asyncio.Event(), asyncio.Event()
        original_operation = service.foreground_operation
        original_info = video_routes.get_video_info

        def persist():
            self.block_worker()
            write_session_metadata()

        @contextmanager
        def observed_operation():
            self.loop.call_soon_threadsafe(second_entered.set)
            with original_operation():
                yield

        def observed_info():
            self.loop.call_soon_threadsafe(read_entered.set)
            return original_info()

        with patch.object(service, "write_session_metadata", side_effect=persist):
            first = self.start_create()
            try:
                await asyncio.wait_for(self.entered.wait(), timeout=1)
                with (patch.object(service, "foreground_operation", observed_operation),
                      patch.object(video_routes, "get_video_info", observed_info)):
                    second = self.start_create()
                    snapshot = self.start_request("GET", "/video/info")
                    await asyncio.wait_for(asyncio.gather(second_entered.wait(), read_entered.wait()), timeout=1)
                    await self.assert_control_requests_respond()
                    self.assertFalse(second.done())
                    self.assertFalse(snapshot.done())
                    persisted = json.loads((self.session_path / "session.json").read_text())
                    self.assertEqual(persisted["annotations"]["revision"], 1)
                    self.release.set()
                    responses = await asyncio.wait_for(asyncio.gather(first, second, snapshot), timeout=5)
            finally:
                self.release.set()
                await asyncio.wait_for(first, timeout=5)
        self.assertEqual([response.status_code for response in responses], [200, 409, 200])
        self.assertEqual(responses[2].json()["annotations"]["revision"], 2)
        self.assertEqual(state.annotations.next_object_id, 3)

    async def test_control_requests_respond_when_worker_pool_is_saturated_by_lock_waits(self):
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 2
        read_entered = asyncio.Event()
        original_info = video_routes.get_video_info

        def persist():
            self.block_worker()
            write_session_metadata()

        def observed_info():
            self.loop.call_soon_threadsafe(read_entered.set)
            return original_info()

        try:
            with (patch.object(service, "write_session_metadata", side_effect=persist),
                  patch.object(video_routes, "get_video_info", observed_info)):
                first = self.start_create()
                await asyncio.wait_for(self.entered.wait(), timeout=1)
                snapshot = self.start_request("GET", "/video/info")
                await asyncio.wait_for(read_entered.wait(), timeout=1)
                self.assertEqual(limiter.borrowed_tokens, 2)
                await self.assert_control_requests_respond()
                self.assertFalse(snapshot.done())
                self.release.set()
                responses = await asyncio.wait_for(asyncio.gather(first, snapshot), timeout=5)
                self.assertEqual([response.status_code for response in responses], [200, 200])
        finally:
            self.release.set()
            await asyncio.wait_for(asyncio.gather(*self.requests, return_exceptions=True), timeout=5)
            limiter.total_tokens = previous_tokens

    async def test_disconnected_request_does_not_release_mutation_lock_before_commit(self):
        def persist():
            self.block_worker()
            write_session_metadata()

        with patch.object(service, "write_session_metadata", side_effect=persist):
            edit = self.start_create()
            try:
                await asyncio.wait_for(self.entered.wait(), timeout=1)
                edit.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await edit
                snapshot = self.start_request("GET", "/video/info")
                await self.assert_control_requests_respond()
                self.assertFalse(snapshot.done())
            finally:
                self.release.set()
            # The HTTP waiter was cancelled; the synchronous edit still owns
            # its lock through persistence, so the snapshot observes acceptance.
            response = await asyncio.wait_for(snapshot, timeout=5)
        self.assertEqual(response.status_code, 200)
        persisted = json.loads((self.session_path / "session.json").read_text())
        self.assertEqual(persisted["annotations"]["revision"], 2)
