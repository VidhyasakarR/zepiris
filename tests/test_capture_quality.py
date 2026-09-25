"""CaptureQualityService rules (stubbed detector and blur model) + the API route."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import quality as quality_routes
from zepiris.deps import MLAsyncDep, S3FetcherDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.ml_inference.capture_quality import CaptureQualityService
from zepiris.schemas.ml_inference import CaptureQualityResult, FaceDetectionResult


class _Face:
    def __init__(self, bbox):
        self.bbox = bbox

    def detect_box(self, img):
        if self.bbox is None:
            return FaceDetectionResult(face_detected=False, bbox=[0, 0, 0, 0])
        return FaceDetectionResult(face_detected=True, bbox=self.bbox, score=0.9)


class _Blur:
    """Returns face_p for the first crop (face) and shirt_p for the second (T-shirt)."""

    def __init__(self, face_p, shirt_p):
        self.ps = [face_p, shirt_p]
        self.calls = 0

    def preprocess(self, crop):
        return crop

    def predict(self, _):
        p = self.ps[min(self.calls, 1)]
        self.calls += 1
        return np.array([p], np.float32)


def _img(level=150, size=(800, 600)):
    return np.full((size[0], size[1], 3), level, np.uint8)


GOOD_FACE = [0.35, 0.10, 0.65, 0.30]  # face 30% of width, T-shirt region fully in frame


def _check(bbox=GOOD_FACE, face_p=0.1, shirt_p=0.2, level=150):
    return CaptureQualityService(_Face(bbox), _Blur(face_p, shirt_p)).check(_img(level))


def test_good_photo_has_no_issues():
    r = _check()
    assert r.issues == [] and r.face_detected and r.shirt_visible == 1.0


def test_no_face():
    assert _check(bbox=None).issues == ["face_not_found"]


def test_blurry_face():
    assert _check(face_p=0.8).issues == ["face_blurry"]


def test_dark_face_reports_dark_not_blur():
    r = _check(face_p=0.9, shirt_p=0.99, level=20)
    assert r.issues == ["too_dark"]


def test_overexposed_face():
    assert _check(level=245).issues == ["too_bright"]


def test_face_too_far():
    assert "face_too_small" in _check(bbox=[0.46, 0.1, 0.54, 0.2]).issues


def test_shirt_cut_off():
    # face low in the frame: the chin-to-stomach region runs off the bottom
    assert "shirt_not_visible" in _check(bbox=[0.35, 0.6, 0.65, 0.85]).issues


def test_shirt_blur_uses_the_stricter_bar():
    assert _check(shirt_p=0.9).issues == []           # plain fabric often scores ~0.9
    assert _check(shirt_p=0.97).issues == ["shirt_blurry"]


# -- API route -----------------------------------------------------------------


class _ML:
    def __init__(self, result):
        self.result = result

    async def check_quality(self, image):
        return self.result


class _Fetcher:
    async def fetch(self, url, *, cacheable=False):
        return b""


def _client(result):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(quality_routes.router, prefix="/v1/quality")
    app.dependency_overrides[MLAsyncDep.__metadata__[0].dependency] = lambda: _ML(result)
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: _Fetcher()
    return TestClient(app)


def _b64():
    ok, buf = cv2.imencode(".jpg", _img())
    return base64.b64encode(buf.tobytes()).decode()


def test_route_ok():
    body = _client(CaptureQualityResult(face_detected=True, issues=[])).post(
        "/v1/quality/check", json={"image_b64": _b64()}).json()
    assert body["ok"] is True and body["warnings"] == []


def test_route_warnings_are_rider_facing():
    body = _client(CaptureQualityResult(face_detected=True, issues=["face_blurry", "shirt_not_visible"])).post(
        "/v1/quality/check", json={"image_b64": _b64()}).json()
    assert body["ok"] is False
    assert [(w["code"], w["region"]) for w in body["warnings"]] == [("face_blurry", "face"), ("shirt_not_visible", "shirt")]
    assert all(w["message"] for w in body["warnings"])
