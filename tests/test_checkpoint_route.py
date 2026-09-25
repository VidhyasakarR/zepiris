"""Route tests for POST /v1/checkpoint/verify — the caller chooses the checks."""

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
    dresscode_threshold = None
    liveness_enabled = False
    allow_threshold_override = False


class _ML:
    def __init__(self, *, color=0.9, logo=0.8, live=True, heads=True) -> None:
        self._dress = DresscodeCheckResult(
            face_detected=True, region="torso", blue_coverage=0.7, logo_match=0.5, score=0.9,
            uniform_score=0.9, engine="siglip2", recommended_threshold=0.35,
            check_scores={"dress_color": color, "logo": logo} if heads else {},
            check_thresholds={"dress_color": 0.38, "logo": 0.40} if heads else {},
        )
        self._live = live
        self.dress_calls = 0

    async def check_dresscode(self, image: bytes) -> DresscodeCheckResult:
        self.dress_calls += 1
        return self._dress

    async def check_liveness(self, image: bytes) -> SpoofDetectionResult:
        return SpoofDetectionResult(is_live=self._live, probability=0.9 if self._live else 0.1)


class _CountingEmbedding(_Embedding):
    calls = 0

    def embed(self, image_rgb):
        type(self).calls += 1
        return super().embed(image_rgb)


def _client(ml: _ML, *, ref_vec=_VEC, liveness=False, overrides=False) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(checkpoint_routes.router, prefix="/v1/checkpoint")
    settings = _Settings()
    settings.liveness_enabled = liveness
    settings.allow_threshold_override = overrides
    app.state.ml_async = ml
    embedding = _CountingEmbedding(live_vec=_VEC, ref_vec=ref_vec)
    _CountingEmbedding.calls = 0
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: settings
    app.dependency_overrides[MatcherDep.__metadata__[0].dependency] = lambda: LocalFaceMatcher(embedding)
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: _Fetcher(_jpeg_bytes())
    app.dependency_overrides[LearnerDep.__metadata__[0].dependency] = lambda: _Learner()
    return TestClient(app)


def _post(client, **body):
    base = {"face_check_b64": _jpeg_b64(), "source_selfie_b64": _jpeg_b64()}
    return client.post("/v1/checkpoint/verify", json={**base, **body})


def test_default_runs_all_three_and_clears() -> None:
    body = _post(_client(_ML())).json()
    assert body["checksRequested"] == ["face_match", "dress_color", "logo"]
    assert set(body["checks"]) == {"face_match", "dress_color", "logo"}
    assert all(c["passed"] for c in body["checks"].values())
    assert body["isCleared"] is True
    assert body["checks"]["dress_color"] == {"passed": True, "score": 0.9, "threshold": 0.38}


def test_one_failing_check_blocks() -> None:
    body = _post(_client(_ML(logo=0.1))).json()
    assert body["checks"]["logo"]["passed"] is False
    assert body["checks"]["face_match"]["passed"] is True
    assert body["isCleared"] is False


def test_face_only_skips_dress_code_entirely() -> None:
    ml = _ML(color=0.0, logo=0.0)  # would fail if it ran
    body = _post(_client(ml), checks=["face_match"]).json()
    assert list(body["checks"]) == ["face_match"]
    assert body["isCleared"] is True
    assert ml.dress_calls == 0
    assert "dresscode" not in body


def test_dress_only_needs_no_source_and_skips_face_match() -> None:
    client = _client(_ML())
    r = client.post("/v1/checkpoint/verify", json={"face_check_b64": _jpeg_b64(), "checks": ["dress_color", "logo"]})
    body = r.json()
    assert r.status_code == 200
    assert set(body["checks"]) == {"dress_color", "logo"}
    assert "faceMatch" not in body
    assert _CountingEmbedding.calls == 0
    assert body["isCleared"] is True


def test_face_and_dress_color_ignores_logo() -> None:
    body = _post(_client(_ML(logo=0.0)), checks=["face_match", "dress_color"]).json()
    assert set(body["checks"]) == {"face_match", "dress_color"}
    assert body["isCleared"] is True  # logo would fail, but it was not asked for


def test_face_match_requires_source() -> None:
    r = _client(_ML()).post("/v1/checkpoint/verify", json={"face_check_b64": _jpeg_b64(), "checks": ["face_match"]})
    assert r.status_code == 400


def test_face_mismatch_blocks() -> None:
    body = _post(_client(_ML(), ref_vec=[0.0, 1.0, 0.0])).json()
    assert body["checks"]["face_match"]["passed"] is False
    assert body["isCleared"] is False


def test_liveness_failure_fails_face_match() -> None:
    body = _post(_client(_ML(live=False), liveness=True), checks=["face_match"]).json()
    assert body["checks"]["face_match"]["livenessFailed"] is True
    assert body["checks"]["face_match"]["passed"] is False
    assert body["isCleared"] is False


def test_threshold_overrides_rejected_by_default() -> None:
    # a rider device must not choose its own pass mark (e.g. -1 clears anyone)
    r = _post(_client(_ML(color=0.6, logo=0.6)), dress_color_threshold=0.1)
    assert r.status_code == 403
    assert r.json()["detail"]["fields"] == ["dress_color_threshold"]


def test_threshold_overrides_bounded_even_when_allowed() -> None:
    client = _client(_ML(), overrides=True)
    for bad in (-1, 0, 1.5):
        assert _post(client, threshold=bad).status_code == 422


def test_threshold_overrides() -> None:
    body = _post(_client(_ML(color=0.6, logo=0.6), overrides=True), dress_color_threshold=0.7, logo_threshold=0.5).json()
    assert body["checks"]["dress_color"] == {"passed": False, "score": 0.6, "threshold": 0.7}
    assert body["checks"]["logo"]["passed"] is True


def test_dress_checks_without_model_are_503() -> None:
    r = _post(_client(_ML(heads=False)), checks=["dress_color"])
    assert r.status_code == 503


def test_unknown_or_empty_checks_rejected() -> None:
    client = _client(_ML())
    assert _post(client, checks=["shoes"]).status_code == 422
    assert _post(client, checks=[]).status_code == 422


def test_both_inputs_for_one_side_rejected() -> None:
    assert _post(_client(_ML()), face_check_s3="https://s3/probe.jpg").status_code == 400
