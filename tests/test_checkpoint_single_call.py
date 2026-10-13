"""The checkpoint's single ML call: face match + dress code on one decode.

Two halves: the ML service's ``/v1/checkpoint/check`` route, and the API
checkpoint path that uses it when both a face and a dress check are asked for.
The API tests compare against the two-call path (flag off) on the same fake ML
results, since the point of the change is that nothing observable moves.
"""

from __future__ import annotations

import cv2
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_checkpoint_route import _Settings
from tests.test_ml_match_route import _Service
from tests.test_verify_route import _Fetcher, _jpeg_b64, _jpeg_bytes, _Learner
from zepiris.api.routes import checkpoint as checkpoint_routes
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.framing import encode_pair_frame
from zepiris.ml_inference.concurrency import InferenceLimiter
from zepiris.ml_inference.logo_text import LogoOcrBusyError
from zepiris.ml_inference.routes import router as ml_router
from zepiris.schemas.ml_inference import (
    CHECKPOINT_FAILED_CHECK_HEADER,
    CheckpointCheckResult,
    DresscodeCheckResult,
    FaceMatchResult,
    SpoofDetectionResult,
)
from zepiris.services.matching import RemoteFaceMatcher

_MATCH = FaceMatchResult(score=0.8, probe_face_detected=True, reference_face_detected=True)
_DRESS = DresscodeCheckResult(
    face_detected=True, region="torso", blue_coverage=0.7, logo_match=0.5, score=0.9,
    uniform_score=0.9, engine="siglip2", recommended_threshold=0.35,
    check_scores={"dress_color": 0.9, "logo": 0.8},
    check_thresholds={"dress_color": 0.38, "logo": 0.40},
    logo_text={"text": "LOADSHARE", "matched": True, "onBlue": True, "reason": None},
)


# ---- ML service: /v1/checkpoint/check -------------------------------------------


def _large_jpeg() -> bytes:
    # Big enough that a reduced decode is visibly smaller than the full one.
    ok, buf = cv2.imencode(".jpg", np.full((128, 128, 3), 90, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _Dresscode:
    def __init__(self, error: Exception | None = None) -> None:
        self.images: list[np.ndarray] = []
        self._error = error

    def check(self, image_rgb):
        self.images.append(image_rgb)
        if self._error is not None:
            raise self._error
        return _DRESS


class _RecordingService(_Service):
    """``_Service`` that also keeps the probe array it was handed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probes: list[np.ndarray] = []

    def match_pair(self, probe_rgb, reference_rgb, **kwargs):
        self.probes.append(probe_rgb)
        return super().match_pair(probe_rgb, reference_rgb, **kwargs)


def _ml_client(face=None, dress=None, *, reduction=0) -> tuple[TestClient, _RecordingService, _Dresscode]:
    face = face if face is not None else _RecordingService(_MATCH)
    dress = dress if dress is not None else _Dresscode()
    app = FastAPI()
    app.include_router(ml_router)
    app.state.inference_limiter = InferenceLimiter(limit=4, queue_timeout=5.0)
    app.state.parallel_pair_embed = False
    app.state.jpeg_decode_reduction = reduction
    app.state.face_embedding_service = face
    app.state.dresscode_service = dress
    return TestClient(app), face, dress


def _check(client, probe=None, reference=None):
    return client.post(
        "/v1/checkpoint/check",
        content=encode_pair_frame(
            _jpeg_bytes() if probe is None else probe,
            _jpeg_bytes() if reference is None else reference,
        ),
        headers={"Content-Type": "application/octet-stream"},
    )


def test_ml_returns_both_results_from_one_decode() -> None:
    client, face, dress = _ml_client()
    r = _check(client)
    assert r.status_code == 200
    assert CheckpointCheckResult(**r.json()) == CheckpointCheckResult(match=_MATCH, dresscode=_DRESS)
    # one decoded array, handed to both halves
    assert face.probes[0] is dress.images[0]
    assert face.calls[0]["want_probe_sharpness"] is False


def test_ml_shared_array_equals_each_endpoints_own_decode() -> None:
    from zepiris.ml_inference.routes import _decode_base64_image, _decode_image_bytes
    import base64

    raw = _large_jpeg()
    client, face, dress = _ml_client()
    assert _check(client, probe=raw).status_code == 200
    as_match = _decode_image_bytes(raw, field="probe")
    as_dress = _decode_base64_image(base64.b64encode(raw).decode())
    assert np.array_equal(face.probes[0], as_match) and face.probes[0].dtype == as_match.dtype
    assert np.array_equal(dress.images[0], as_dress) and dress.images[0].dtype == as_dress.dtype


def test_ml_reduced_decode_stays_on_the_match_only() -> None:
    client, face, dress = _ml_client(reduction=2)
    assert _check(client, probe=_large_jpeg()).status_code == 200
    assert face.probes[0].shape == (64, 64, 3)
    assert dress.images[0].shape == (128, 128, 3)  # dress code always full resolution


def test_ml_logo_ocr_busy_is_503() -> None:
    client, _, _ = _ml_client(dress=_Dresscode(error=LogoOcrBusyError("all slots taken")))
    r = _check(client)
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "logo_ocr_busy"
    assert CHECKPOINT_FAILED_CHECK_HEADER not in r.headers


def test_ml_undecodable_probe_fails_as_dresscode_check_does() -> None:
    client, face, dress = _ml_client()
    r = _check(client, probe=b"not an image")
    assert r.status_code == 400
    assert r.json()["detail"] == "failed_to_decode_image"
    assert CHECKPOINT_FAILED_CHECK_HEADER not in r.headers
    assert face.calls == [] and dress.images == []


def test_ml_undecodable_reference_is_tagged_face_match() -> None:
    client, _, dress = _ml_client()
    r = _check(client, reference=b"not an image")
    assert r.status_code == 400
    assert "reference" in r.json()["detail"]
    assert r.headers[CHECKPOINT_FAILED_CHECK_HEADER] == "face_match"
    assert dress.images == []


def test_ml_malformed_frame_is_400() -> None:
    client, _, _ = _ml_client()
    r = client.post("/v1/checkpoint/check", content=b"\x00\x00", headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 400
    assert r.json()["detail"].startswith("malformed_frame")


def test_ml_missing_models_are_503() -> None:
    client, _, _ = _ml_client()
    client.app.state.dresscode_service = None
    r = _check(client)
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "dresscode_unavailable"
    assert CHECKPOINT_FAILED_CHECK_HEADER not in r.headers

    client.app.state.face_embedding_service = None
    r = _check(client)
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "face_embedding_unavailable"
    assert r.headers[CHECKPOINT_FAILED_CHECK_HEADER] == "face_match"


# ---- API: /v1/checkpoint/{verify,score} -----------------------------------------


def _status_error(status: int, body: dict, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://ml/x")
    response = httpx.Response(status, json=body, headers=headers, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


class _ML:
    """Fake async ML client: the two-call endpoints and the combined one agree."""

    def __init__(self, *, combined_error=None, match_error=None, dress_error=None) -> None:
        self.calls: list[str] = []
        self._combined_error = combined_error
        self._match_error = match_error
        self._dress_error = dress_error

    async def match_faces(self, probe, reference, *, want_probe_sharpness=False):
        self.calls.append("match")
        if self._match_error is not None:
            raise self._match_error
        return _MATCH

    async def check_dresscode(self, image):
        self.calls.append("dresscode")
        if self._dress_error is not None:
            raise self._dress_error
        return _DRESS

    async def check_checkpoint(self, probe, reference):
        self.calls.append("checkpoint")
        if self._combined_error is not None:
            raise self._combined_error
        return CheckpointCheckResult(match=_MATCH, dresscode=_DRESS)

    async def check_liveness(self, image):
        self.calls.append("liveness")
        return SpoofDetectionResult(is_live=True, probability=0.9)


def _api_client(ml: _ML, *, single_call=True, liveness=False) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(checkpoint_routes.router, prefix="/v1/checkpoint")
    settings = _Settings()
    settings.checkpoint_single_call = single_call
    settings.liveness_enabled = liveness
    app.state.ml_async = ml
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: settings
    app.dependency_overrides[MatcherDep.__metadata__[0].dependency] = lambda: RemoteFaceMatcher(ml)
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: _Fetcher(_jpeg_bytes())
    app.dependency_overrides[LearnerDep.__metadata__[0].dependency] = lambda: _Learner()
    return TestClient(app)


def _post(client, path="verify", **body):
    base = {"face_check_b64": _jpeg_b64(), "source_selfie_b64": _jpeg_b64()}
    return client.post(f"/v1/checkpoint/{path}", json={**base, **body})


def _stable(body: dict) -> dict:
    body = dict(body)
    body.pop("requestId", None)
    body.pop("scoredAt", None)
    return body


def test_api_face_and_dress_make_one_ml_call() -> None:
    ml = _ML()
    r = _post(_api_client(ml))
    assert r.status_code == 200
    assert ml.calls == ["checkpoint"]
    assert r.json()["isCleared"] is True


def test_api_liveness_still_runs_beside_the_single_call() -> None:
    ml = _ML()
    assert _post(_api_client(ml, liveness=True)).status_code == 200
    assert sorted(ml.calls) == ["checkpoint", "liveness"]


def test_api_face_only_and_dress_only_keep_their_own_calls() -> None:
    ml = _ML()
    assert _post(_api_client(ml), checks=["face_match"]).status_code == 200
    assert ml.calls == ["match"]

    ml = _ML()
    r = _api_client(ml).post(
        "/v1/checkpoint/verify", json={"face_check_b64": _jpeg_b64(), "checks": ["logo"]}
    )
    assert r.status_code == 200
    assert ml.calls == ["dresscode"]


def test_api_flag_off_uses_the_two_calls() -> None:
    ml = _ML()
    assert _post(_api_client(ml, single_call=False)).status_code == 200
    assert sorted(ml.calls) == ["dresscode", "match"]


def test_api_responses_identical_with_flag_on_and_off() -> None:
    for path in ("verify", "score"):
        for liveness in (False, True):
            for checks in (["face_match", "dress_color", "logo"], ["face_match", "logo"]):
                on = _post(_api_client(_ML(), liveness=liveness), path, checks=checks)
                off = _post(_api_client(_ML(), single_call=False, liveness=liveness), path, checks=checks)
                assert on.status_code == off.status_code == 200
                assert _stable(on.json()) == _stable(off.json())


def test_api_logo_ocr_busy_is_503_as_before() -> None:
    busy = {"detail": {"message": "logo_ocr_busy", "error": "all slots taken"}}
    on = _post(_api_client(_ML(combined_error=_status_error(503, busy))))
    off = _post(_api_client(_ML(dress_error=_status_error(503, busy)), single_call=False))
    assert on.status_code == off.status_code == 503
    assert on.json() == off.json()
    assert on.json()["detail"]["message"] == "dresscode_upstream_error"


def test_api_face_match_failure_maps_as_the_matcher_does() -> None:
    bad_reference = {"detail": "failed_to_decode_image: reference"}
    on = _post(_api_client(_ML(combined_error=_status_error(
        400, bad_reference, {CHECKPOINT_FAILED_CHECK_HEADER: "face_match"}
    ))))
    off = _post(_api_client(_ML(match_error=_status_error(400, bad_reference)), single_call=False))
    assert on.status_code == off.status_code == 400
    assert on.json() == off.json()


def test_api_ml_timeout_maps_as_the_matcher_does() -> None:
    timeout = httpx.ReadTimeout("slow")
    on = _post(_api_client(_ML(combined_error=timeout)))
    off = _post(_api_client(_ML(match_error=timeout), single_call=False))
    assert on.status_code == off.status_code
    assert on.json() == off.json()
