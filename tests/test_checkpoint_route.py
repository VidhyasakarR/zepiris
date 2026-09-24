"""Route tests for POST /v1/checkpoint/verify (face match + dress code together)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_verify_route import _Embedding, _Fetcher, _jpeg_b64, _jpeg_bytes, _Learner
from zepiris.api.routes import checkpoint as checkpoint_routes
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.schemas.ml_inference import DresscodeCheckResult, SpoofDetectionResult
from zepiris.services.matching import LocalFaceMatcher

_VEC = [1.0, 0.0, 0.0]


class _Settings:
    verify_threshold = 0.5
    dresscode_threshold = 0.55
    liveness_enabled = False


class _ML:
    def __init__(self, *, dress_score=0.8, blue=0.9, logo=0.6, live=True) -> None:
        self._dress = DresscodeCheckResult(
            face_detected=True, region="torso", blue_coverage=blue, logo_match=logo, score=dress_score
        )
        self._live = live
        self.dress_images: list[bytes] = []

    async def check_dresscode(self, image: bytes) -> DresscodeCheckResult:
        self.dress_images.append(image)
        return self._dress

    async def check_liveness(self, image: bytes) -> SpoofDetectionResult:
        return SpoofDetectionResult(is_live=self._live, probability=0.9 if self._live else 0.1)


def _client(ml: _ML, *, ref_vec=_VEC, liveness=False) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(checkpoint_routes.router, prefix="/v1/checkpoint")
    settings = _Settings()
    settings.liveness_enabled = liveness
    app.state.ml_async = ml
    embedding = _Embedding(live_vec=_VEC, ref_vec=ref_vec)
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: settings
    app.dependency_overrides[MatcherDep.__metadata__[0].dependency] = lambda: LocalFaceMatcher(embedding)
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: _Fetcher(_jpeg_bytes())
    app.dependency_overrides[LearnerDep.__metadata__[0].dependency] = lambda: _Learner()
    return TestClient(app)


def _body(**extra) -> dict:
    return {"face_check_b64": _jpeg_b64(), "source_selfie_b64": _jpeg_b64(), **extra}


def test_cleared_when_face_and_uniform_match() -> None:
    ml = _ML()
    body = _client(ml).post("/v1/checkpoint/verify", json=_body()).json()
    assert body["isCleared"] is True
    assert body["scores"]["faceMatch"] == 1.0
    assert body["scores"]["dressColour"] == 0.9
    assert body["scores"]["logo"] == 0.6
    assert body["faceMatch"]["verificationResult"]["isMatch"] is True
    assert body["dresscode"]["isMatch"] is True
    assert "requestId" not in body["faceMatch"] and "requestId" not in body["dresscode"]
    assert ml.dress_images == [_jpeg_bytes()]  # dress code scores the selfie, not the source


def test_not_cleared_when_uniform_fails() -> None:
    body = _client(_ML(dress_score=0.2)).post("/v1/checkpoint/verify", json=_body()).json()
    assert body["faceMatch"]["verificationResult"]["isMatch"] is True
    assert body["dresscode"]["isMatch"] is False
    assert body["isCleared"] is False


def test_not_cleared_when_face_differs() -> None:
    body = _client(_ML(), ref_vec=[0.0, 1.0, 0.0]).post("/v1/checkpoint/verify", json=_body()).json()
    assert body["faceMatch"]["verificationResult"]["isMatch"] is False
    assert body["isCleared"] is False


def test_liveness_failure_blocks_clearance() -> None:
    body = _client(_ML(live=False), liveness=True).post("/v1/checkpoint/verify", json=_body()).json()
    assert body["faceMatch"]["livenessFailed"] is True
    assert body["scores"]["liveness"] == 0.1
    assert body["isCleared"] is False


def test_dresscode_threshold_override() -> None:
    body = _client(_ML(dress_score=0.6)).post(
        "/v1/checkpoint/verify", json=_body(dresscode_threshold=0.7)
    ).json()
    assert body["dresscode"]["threshold"] == 0.7
    assert body["dresscode"]["isMatch"] is False


def test_source_from_s3_url() -> None:
    body = _client(_ML()).post(
        "/v1/checkpoint/verify",
        json={"face_check_b64": _jpeg_b64(), "source_selfie_s3": "https://s3/ref.jpg"},
    ).json()
    assert body["isCleared"] is True


def test_both_inputs_for_one_side_rejected() -> None:
    r = _client(_ML()).post(
        "/v1/checkpoint/verify",
        json=_body(face_check_s3="https://s3/probe.jpg"),
    )
    assert r.status_code == 400
