"""Route tests for POST /v1/checkpoint/verify — the caller chooses the checks."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_verify_route import _Embedding, _Fetcher, _jpeg_b64, _jpeg_bytes, _Learner
from zepiris.api.routes import checkpoint as checkpoint_routes
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.schemas.ml_inference import CaptureQualityResult, DresscodeCheckResult, SpoofDetectionResult
from zepiris.services.matching import LocalFaceMatcher

_VEC = [1.0, 0.0, 0.0]


class _Settings:
    verify_threshold = 0.5
    dresscode_threshold = None
    liveness_enabled = False
    allow_threshold_override = False
    checkpoint_quality_gate = False


class _ML:
    def __init__(self, *, color=0.9, logo=0.8, live=True, heads=True, issues=()) -> None:
        self._dress = DresscodeCheckResult(
            face_detected=True, region="torso", blue_coverage=0.7, logo_match=0.5, score=0.9,
            uniform_score=0.9, engine="siglip2", recommended_threshold=0.35,
            check_scores={"dress_color": color, "logo": logo} if heads else {},
            check_thresholds={"dress_color": 0.38, "logo": 0.40} if heads else {},
        )
        self._live = live
        self._issues = list(issues)
        self.dress_calls = 0
        self.liveness_calls = 0
        self.quality_calls = 0

    async def check_dresscode(self, image: bytes) -> DresscodeCheckResult:
        self.dress_calls += 1
        return self._dress

    async def check_quality(self, image: bytes) -> CaptureQualityResult:
        self.quality_calls += 1
        return CaptureQualityResult(face_detected=True, face_width=0.4, face_brightness=120.0,
                                    face_blur=0.1, shirt_visible=0.9, shirt_blur=0.1, issues=self._issues)

    async def check_liveness(self, image: bytes) -> SpoofDetectionResult:
        self.liveness_calls += 1
        return SpoofDetectionResult(is_live=self._live, probability=0.9 if self._live else 0.1)


class _CountingEmbedding(_Embedding):
    calls = 0

    def embed(self, image_rgb):
        type(self).calls += 1
        return super().embed(image_rgb)


def _client(ml: _ML, *, ref_vec=_VEC, liveness=False, overrides=False, gate=False) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(checkpoint_routes.router, prefix="/v1/checkpoint")
    settings = _Settings()
    settings.liveness_enabled = liveness
    settings.allow_threshold_override = overrides
    settings.checkpoint_quality_gate = gate
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


# ---- /score: scores only, no decision ---------------------------------------------
import base64 as _b64  # noqa: E402
import hashlib as _hashlib  # noqa: E402


def _score(client, **body):
    base = {"face_check_b64": _jpeg_b64(), "source_selfie_b64": _jpeg_b64()}
    return client.post("/v1/checkpoint/score", json={**base, **body})


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_score_has_no_decision_anywhere() -> None:
    body = _score(_client(_ML(color=0.6, logo=0.2))).json()
    keys = set(_walk_keys(body))
    assert not keys & {"passed", "isCleared", "threshold", "isMatch", "livenessFailed", "signature"}
    assert set(body) == {"requestId", "scoredAt", "checksRequested", "scores", "image"}
    assert body["checksRequested"] == ["face_match", "dress_color", "logo"]
    assert body["scores"]["dress_color"] == {"score": 0.6}
    assert body["scores"]["logo"] == {"score": 0.2}
    fm = body["scores"]["face_match"]
    assert set(fm) == {"similarity", "liveness", "faceDetected"} and fm["faceDetected"] is True


def test_score_reports_liveness_even_when_not_live() -> None:
    fm = _score(_client(_ML(live=False), liveness=True), checks=["face_match"]).json()["scores"]["face_match"]
    assert fm["liveness"] == 0.1 and fm["similarity"] is not None


def test_score_only_runs_requested_checks() -> None:
    body = _client(_ML()).post("/v1/checkpoint/score", json={"face_check_b64": _jpeg_b64(), "checks": ["logo"]}).json()
    assert set(body["scores"]) == {"logo"}


def test_score_image_hash_is_of_the_exact_photo() -> None:
    photo = _jpeg_b64()
    body = _client(_ML()).post("/v1/checkpoint/score", json={"face_check_b64": photo, "checks": ["logo"]}).json()
    raw = _b64.b64decode(photo)
    assert body["image"] == {"sha256": _hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


# ---- quality gate ---------------------------------------------------------


def test_gate_passes_then_runs_face_and_dress() -> None:
    ml = _ML()
    body = _post(_client(ml, gate=True, liveness=True)).json()
    assert body["isCleared"] is True
    assert set(body["checks"]) == {"quality", "liveness", "face_match", "dress_color", "logo"}
    assert body["checks"]["quality"]["passed"] is True
    assert body["checks"]["liveness"] == {"passed": True, "score": 0.9, "reason": None}
    # the gate's liveness result is reused by face match, not fetched again
    assert ml.liveness_calls == 1
    assert body["checks"]["face_match"]["liveness"] == 0.9


def test_blurry_face_stops_before_face_match_and_dress() -> None:
    ml = _ML(issues=["face_blurry"])
    body = _post(_client(ml, gate=True)).json()
    assert body["isCleared"] is False
    assert body["failedAt"] == "quality_gate"
    assert body["checks"]["quality"]["issues"] == ["face_blurry"]
    assert body["checks"]["quality"]["warnings"][0]["region"] == "face"
    assert "face_match" not in body["checks"] and "faceMatch" not in body
    assert ml.dress_calls == 0
    assert _CountingEmbedding.calls == 0


def test_spoof_stops_before_face_match() -> None:
    ml = _ML(live=False)
    body = _post(_client(ml, gate=True, liveness=True)).json()
    assert body["isCleared"] is False
    assert body["failedAt"] == "quality_gate"
    assert body["checks"]["liveness"]["passed"] is False
    assert ml.dress_calls == 0


def test_shirt_issues_ignored_when_no_dress_check() -> None:
    ml = _ML(issues=["shirt_not_visible"])
    body = _post(_client(ml, gate=True), checks=["face_match"]).json()
    assert body["checks"]["quality"] == {**body["checks"]["quality"], "passed": True, "issues": []}
    assert body["isCleared"] is True


def test_gate_off_skips_quality_and_liveness_calls() -> None:
    ml = _ML(issues=["face_blurry"])
    body = _post(_client(ml)).json()
    assert ml.quality_calls == 0
    assert "quality" not in body["checks"]


def test_quality_check_false_skips_quality_but_keeps_liveness() -> None:
    ml = _ML(issues=["face_blurry"])
    body = _post(_client(ml, gate=True, liveness=True), quality_check=False).json()
    assert ml.quality_calls == 0
    assert "quality" not in body["checks"]
    assert body["checks"]["liveness"]["passed"] is True
    assert body["isCleared"] is True


def test_quality_check_false_and_no_liveness_skips_the_gate() -> None:
    ml = _ML(issues=["face_blurry"])
    body = _post(_client(ml, gate=True), quality_check=False).json()
    assert ml.quality_calls == 0
    assert "failedAt" not in body and body["isCleared"] is True


def test_unreadable_selfie_with_quality_off_is_still_a_retake() -> None:
    from zepiris.services.matching import ProbeImageDecodeError

    class _BadImageML(_ML):
        async def check_liveness(self, image: bytes):
            raise ProbeImageDecodeError("failed_to_decode_image: probe")

    ml = _BadImageML()
    r = _post(_client(ml, gate=True, liveness=True), quality_check=False)
    assert r.status_code == 200
    body = r.json()
    assert body["failedAt"] == "quality_gate"
    assert body["checks"]["quality"]["issues"] == ["image_unreadable"]
    assert ml.dress_calls == 0


def test_unreadable_selfie_is_a_retake_not_an_error() -> None:
    import httpx

    class _BadImageML(_ML):
        async def check_quality(self, image: bytes):
            req = httpx.Request("POST", "http://ml/v1/quality/check")
            raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(400, request=req))

    ml = _BadImageML()
    r = _post(_client(ml, gate=True, liveness=True))
    assert r.status_code == 200
    body = r.json()
    assert body["isCleared"] is False
    assert body["checks"]["quality"]["issues"] == ["image_unreadable"]
    assert ml.dress_calls == 0
