"""The API contract between the backend and the frontend, checked mechanically.

Every route declares what it returns, and every field the frontend's type file names exists in the
backend model it corresponds to, so a rename on either side fails a test instead of failing in the UI.
"""
import re
import sys
import unittest
from pathlib import Path

from fastapi.routing import APIRoute

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.factory import create_app
from schemas import responses
from schemas.tracking import TrackingPromptPointsRequest
from schemas.video import (
    AnnotationPrecondition,
    InteractiveState,
    MaskOverride,
    VideoAddMaskRequest,
    VideoAddPointsOrBoxRequest,
    VideoInitStateRequest,
    VideoObject,
    VideoPrompt,
    VideoPropagateRequest,
    VideoSaveRequest,
)
from support import BackendStateTestCase

FRONTEND_TYPES = Path(__file__).resolve().parents[2] / "frontend-ng/data-engine/src/app/services/backend-api.types.ts"

# Routes that stream a file instead of returning JSON.
FILE_ROUTES = {"/video/frame/{frame_idx}", "/video/mask_frame/{frame_idx}"}

# TypeScript interface -> the backend model(s) it describes. The frontend declares only what it reads, and
# reuses a few interfaces for several responses, so a field must exist in at least one of the models.
CORRESPONDING_MODELS = {
    "VideoInitStateRequest": [VideoInitStateRequest],
    "VideoAddPointsOrBoxRequest": [VideoAddPointsOrBoxRequest],
    "VideoPropagateRequest": [VideoPropagateRequest],
    "VideoAddMaskRequest": [VideoAddMaskRequest],
    "VideoSaveRequest": [VideoSaveRequest],
    "VideoSaveInteractiveState": [InteractiveState],
    "TrackPromptPointsRequest": [TrackingPromptPointsRequest],
    "AnnotationPrecondition": [AnnotationPrecondition],
    "SavedState": [responses.SavedState],
    "VideoSaveResponse": [responses.SaveResponse],
    "VideoAddPointsResponse": [responses.PointsResponse],
    "VideoInitStateResponse": [responses.InitResult],
    "VideoObject": [VideoObject],
    "MaskOverride": [MaskOverride],
    "VideoPrompt": [VideoPrompt],
    "VideoAnnotations": [responses.AnnotationsSnapshot],
    "AnnotationDelta": [responses.AnnotationDelta],
    "TrackingValidity": [responses.TrackingValidity],
    "AnnotationSnapshot": [responses.VideoInfoResponse, responses.AnnotationUpdate, responses.MaskEditResponse],
    "AnnotationUpdate": [responses.AnnotationUpdate, responses.MaskEditResponse],
    "ObjectCreationResponse": [responses.ObjectCreationResponse],
    "VideoInfoResponse": [responses.VideoInfoResponse],
    "RestoredSessionPayload": [responses.RestoredSession],
    "VideoPropagateResponse": [responses.PropagationResult],
    "VideoMaskObjectData": [responses.RleMask],
    "PackedBitMask": [responses.PackedMask],
    "VideoMaskDataResponse": [responses.MaskData],
    "TrackPromptPointMetadata": [responses.TrackedPoint],
    "TrackPromptPointsJobResponse": [responses.TrackingJobResult],
    "TrackPromptPointsResult": [responses.TrackingResult],
    "HealthResponse": [responses.HealthResponse],
    "JobStartResponse": [responses.JobStartResponse],
    "JobError": [responses.JobError],
    "JobStageHistoryEntry": [responses.JobStageEntry],
    "BackendJob": [responses.JobSnapshot],
}


def typescript_interfaces(source: str) -> dict[str, set[str]]:
    """The fields each exported interface declares itself (inherited ones are checked on the parent)."""
    interfaces = {}
    for match in re.finditer(r"export interface (\w+)(?:<[^>]*>)?(?: extends \w+)? \{\n(.*?)\n\}", source, re.S):
        interfaces[match.group(1)] = set(re.findall(r"^  ['\"]?(\w+)['\"]?\??:", match.group(2), re.M))
    return interfaces


class ApiContractTests(BackendStateTestCase):
    def test_every_json_route_declares_what_it_returns(self):
        routes = [route for route in create_app(self.ctx).routes if isinstance(route, APIRoute)]
        self.assertGreater(len(routes), 15)
        undeclared = [f"{sorted(route.methods)} {route.path}" for route in routes
                      if route.response_model is None and route.path not in FILE_ROUTES]
        self.assertEqual(undeclared, [], "Routes without a response model: " + ", ".join(undeclared))

    @unittest.skipUnless(FRONTEND_TYPES.is_file(), "the frontend source is not next to the backend")
    def test_every_field_the_frontend_names_exists_in_the_backend_models(self):
        interfaces = typescript_interfaces(FRONTEND_TYPES.read_text(encoding="utf-8"))
        self.assertGreater(len(interfaces), 25)
        missing_pairs = sorted(set(CORRESPONDING_MODELS) - set(interfaces))
        self.assertEqual(missing_pairs, [], f"The frontend no longer declares: {missing_pairs}")
        problems = []
        for name, models in CORRESPONDING_MODELS.items():
            known = set().union(*(model.model_fields for model in models))
            unknown = sorted(interfaces[name] - known)
            if unknown:
                problems.append(f"{name} names {unknown}, which {[m.__name__ for m in models]} do not have")
        self.assertEqual(problems, [], "Frontend fields with no backend counterpart:\n" + "\n".join(problems))

    def test_the_contract_test_would_notice_a_missing_field(self):
        interfaces = typescript_interfaces("export interface Example<T = unknown> extends Base {\n  kept: number;\n  renamed?: string;\n}\n")
        self.assertEqual(interfaces, {"Example": {"kept", "renamed"}})
        self.assertEqual(sorted(interfaces["Example"] - set(responses.SavedState.model_fields)), ["kept", "renamed"])


if __name__ == "__main__":
    unittest.main()
