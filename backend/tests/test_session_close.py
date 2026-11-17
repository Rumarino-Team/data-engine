import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schemas.video import VideoInitStateRequest
from support import SessionStateTestCase, state
from video import loading, prompts


class SessionCloseTests(SessionStateTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()
        cache = patch("video.io.SESSION_CACHE_ROOT", self.root / "sessions")
        cache.start()
        self.addCleanup(cache.stop)
        self.masker = self.make_masker()
        self.masker.state_loaded = True
        self.masker.release_state.side_effect = lambda: setattr(self.masker, "state_loaded", False)
        self.working = self.working_session(
            self.root / "sessions" / "open", session_id="open", video_masker=self.masker)
        state.annotations.create_object()
        self.open_session = self.ctx.session
        self.client = self.make_client()

    def assert_still_open(self):
        self.assertIs(self.ctx.session, self.open_session)
        self.assertEqual(self.ctx.session.active_session_id, "open")
        self.assertTrue(self.working.exists())
        self.masker.release_state.assert_not_called()

    def test_closing_frees_the_video_and_its_working_copy_and_keeps_only_the_weights(self):
        response = self.client.post("/video/close_session")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Session closed", "closed_session_id": "open"})
        self.masker.release_state.assert_called_once()
        closed = self.ctx.session
        self.assertIsNot(closed, self.open_session)
        self.assertIsNone(closed.active_session_id)
        self.assertIsNone(closed.video_dir)
        self.assertEqual(closed.video_frame_files, [])
        self.assertEqual(closed.annotations.objects, {})
        self.assertIsNone(closed.tracker)
        self.assertFalse(self.working.exists())
        self.assertIs(closed.video_masker, self.masker)  # the weights; its video state was released

    def test_a_closed_session_answers_like_a_backend_with_nothing_loaded(self):
        stale_headers = {"X-Session-ID": "open", "X-Annotation-Revision": str(state.annotations.revision)}
        self.assertEqual(self.client.post("/video/close_session").status_code, 200)

        info = self.client.get("/video/info").json()
        self.assertEqual((info["num_frames"], info["predictor_ready"], info["session_id"]), (0, False, None))
        self.assertEqual(self.client.get("/video/frame/0").status_code, 400)
        self.assertEqual(self.client.get("/video/mask_data/0").status_code, 400)
        # Edits that still name the closed session are refused instead of acting on whatever comes next.
        edit = self.client.post("/video/objects", headers=stale_headers)
        self.assertEqual(edit.status_code, 409)
        again = self.client.post("/video/close_session", headers=stale_headers)
        self.assertEqual(again.status_code, 409)

    def test_closing_names_the_session_and_revision_and_refuses_a_stale_one(self):
        for headers in ({"X-Session-ID": "another"}, {"X-Annotation-Revision": "999"}):
            with self.subTest(headers=headers):
                response = self.client.post("/video/close_session", headers=headers)
                self.assertEqual(response.status_code, 409)
                self.assert_still_open()


if __name__ == "__main__":
    unittest.main()
