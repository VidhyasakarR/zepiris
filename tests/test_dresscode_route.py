"""Route tests for POST /v1/dresscode/match.

The ML client is stubbed: these tests cover input resolution, threshold
application, response shape and upstream error mapping, not detection quality.
"""

from __future__ import annotations

import base64

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import dresscode as dresscode_routes
from zepiris.deps import MLAsyncDep, S3FetcherDep, SettingsDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.schemas.face import MAX_IMAGE_SIZE_BYTES
from zepiris.schemas.ml_inference import DresscodeCheckResult


def _jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _jpeg_b64() -> str:
    return base64.b64encode(_jpeg_bytes()).decode()


class _Settings:
    dresscode_threshold = 0.55


class _Fetcher:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.calls: list[tuple[str, bool]] = []

    async def fetch(self, url: str, *, cacheable: bool = False) -> bytes:
        self.calls.append((url, cacheable))
        return self._data


class _ML:
    """Stub ML client returning a canned result, or raising a canned error."""

    def __init__(self, result: DresscodeCheckResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.received: list[bytes] = []

    async def check_dresscode(self, image: bytes) -> DresscodeCheckResult:
        self.received.append(image)
        if self._error is not None:
            raise self._error
        return self._result


def _result(**kwargs) -> DresscodeCheckResult:
    base = {
        "face_detected": True,
        "region": "torso",
        "blue_coverage": 0.89,
        "logo_match": 0.61,
        "score": 0.81,
        "reason": None,
    }
    return DresscodeCheckResult(**{**base, **kwargs})


def _client(ml: _ML, fetcher: _Fetcher | None = None, settings=None) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(dresscode_routes.router, prefix="/v1/dresscode")
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: settings or _Settings()
    app.dependency_overrides[MLAsyncDep.__metadata__[0].dependency] = lambda: ml
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: (
        fetcher or _Fetcher(_jpeg_bytes())
    )
    return TestClient(app)


# -- happy paths -------------------------------------------------------------


def test_base64_returns_documented_shape():
    client = _client(_ML(_result()))
    resp = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["isMatch"] is True
    assert body["score"] == pytest.approx(0.81)
    assert body["threshold"] == pytest.approx(0.55)
    assert body["scores"] == {
        "uniform": None, "blueCoverage": pytest.approx(0.89), "logoMatch": pytest.approx(0.61)
    }
    assert body["engine"] == "hsv"
    assert body["faceDetected"] is True
    assert body["region"] == "torso"
    assert body["reason"] is None
    assert body["requestId"]


def test_data_uri_base64_is_accepted():
    client = _client(_ML(_result()))
    resp = client.post(
        "/v1/dresscode/match", json={"image_b64": f"data:image/jpeg;base64,{_jpeg_b64()}"}
    )
    assert resp.status_code == 200


def test_s3_url_is_fetched_and_not_cached():
    """A dress-code probe is a one-shot capture; caching it would only churn."""
    fetcher = _Fetcher(_jpeg_bytes())
    ml = _ML(_result())
    client = _client(ml, fetcher)
    resp = client.post(
        "/v1/dresscode/match", json={"image_s3": "https://bucket.s3.amazonaws.com/a.jpg"}
    )
    assert resp.status_code == 200
    assert fetcher.calls == [("https://bucket.s3.amazonaws.com/a.jpg", False)]
    assert ml.received == [_jpeg_bytes()]


def test_score_below_threshold_is_not_a_match():
    client = _client(_ML(_result(score=0.30)))
    body = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()}).json()
    assert body["isMatch"] is False


def test_explicit_threshold_overrides_default():
    client = _client(_ML(_result(score=0.60)))
    body = client.post(
        "/v1/dresscode/match", json={"image_b64": _jpeg_b64(), "threshold": 0.90}
    ).json()
    assert body["threshold"] == pytest.approx(0.90)
    assert body["isMatch"] is False


def test_null_logo_match_is_preserved_not_zeroed():
    """`null` means "not measured" and must not be reported as a zero score."""
    client = _client(_ML(_result(logo_match=None, score=0.89)))
    body = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()}).json()
    assert body["scores"]["logoMatch"] is None


def test_unscored_reason_never_reports_a_match():
    """A high score is meaningless when the region could not be scored."""
    client = _client(_ML(_result(reason="roi_too_small", score=0.99)))
    body = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()}).json()
    assert body["isMatch"] is False
    assert body["reason"] == "roi_too_small"


def test_no_face_reports_fallback_region():
    client = _client(_ML(_result(face_detected=False, region="fallback_full_image")))
    body = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()}).json()
    assert body["faceDetected"] is False
    assert body["region"] == "fallback_full_image"


# -- input validation --------------------------------------------------------


def test_both_inputs_is_ambiguous():
    client = _client(_ML(_result()))
    resp = client.post(
        "/v1/dresscode/match", json={"image_b64": _jpeg_b64(), "image_s3": "https://x/y.jpg"}
    )
    assert resp.status_code == 400


def test_neither_input_is_missing():
    client = _client(_ML(_result()))
    assert client.post("/v1/dresscode/match", json={}).status_code == 400


def test_invalid_base64_is_rejected():
    client = _client(_ML(_result()))
    assert client.post("/v1/dresscode/match", json={"image_b64": "!!!not base64!!!"}).status_code == 400


def test_empty_image_is_rejected():
    client = _client(_ML(_result()))
    resp = client.post("/v1/dresscode/match", json={"image_b64": base64.b64encode(b"").decode()})
    assert resp.status_code == 400


def test_oversize_image_is_rejected_with_422():
    oversize = base64.b64encode(b"\x00" * (MAX_IMAGE_SIZE_BYTES + 1)).decode()
    client = _client(_ML(_result()))
    assert client.post("/v1/dresscode/match", json={"image_b64": oversize}).status_code == 422


# -- upstream failures -------------------------------------------------------


def test_unreachable_ml_service_returns_503():
    client = _client(_ML(error=httpx.ConnectError("refused")))
    resp = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()})
    assert resp.status_code == 503


def test_ml_service_error_status_is_propagated():
    request = httpx.Request("POST", "http://ml/v1/dresscode/check")
    response = httpx.Response(503, text="dresscode_unavailable", request=request)
    client = _client(_ML(error=httpx.HTTPStatusError("x", request=request, response=response)))
    resp = client.post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()})
    assert resp.status_code == 503


def test_classifier_engine_uses_its_calibrated_threshold() -> None:
    """With no request/settings threshold, the learned engine's own threshold applies."""
    result = DresscodeCheckResult(
        face_detected=True, region="torso", blue_coverage=0.2, logo_match=None,
        score=0.4, uniform_score=0.4, engine="siglip2", recommended_threshold=0.35,
    )
    ml = _ML(result=result)
    settings = _Settings()
    settings.dresscode_threshold = None
    body = _client(ml, settings=settings).post("/v1/dresscode/match", json={"image_b64": _jpeg_b64()}).json()
    assert body["engine"] == "siglip2"
    assert body["threshold"] == pytest.approx(0.35)
    assert body["isMatch"] is True
    assert body["scores"]["uniform"] == pytest.approx(0.4)
