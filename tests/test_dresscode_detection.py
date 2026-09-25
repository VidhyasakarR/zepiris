"""Unit tests for DresscodeDetectionService.

The face detector is stubbed throughout: these tests exercise the ROI geometry,
the colour band and the fusion rules, none of which need a loaded model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from zepiris.ml_inference.dresscode_detection import DresscodeDetectionService
from zepiris.schemas.ml_inference import FaceDetectionResult


class _FakeFaceService:
    """Stands in for FaceEmbeddingService; only detect_box is used."""

    def __init__(self, bbox: list[float] | None) -> None:
        self._bbox = bbox

    def detect_box(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        if self._bbox is None:
            return FaceDetectionResult(face_detected=False, bbox=[0.0, 0.0, 0.0, 0.0])
        return FaceDetectionResult(face_detected=True, bbox=self._bbox, score=0.9)


def _swatch(h: int, s: int, v: int, size: tuple[int, int] = (400, 300)) -> np.ndarray:
    """Solid HSV swatch returned as RGB."""
    hsv = np.full((size[0], size[1], 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _service(bbox: list[float] | None = None, **kwargs) -> DresscodeDetectionService:
    return DresscodeDetectionService(_FakeFaceService(bbox), **kwargs)


# -- colour band -------------------------------------------------------------


def test_brand_blue_scores_full_coverage():
    """Shirt-fabric blue measured from the reference photos is inside the band."""
    result = _service().check(_swatch(113, 241, 169))
    assert result.blue_coverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "hsv"),
    [("red", (0, 240, 180)), ("grey", (0, 0, 128)), ("green", (60, 200, 160))],
)
def test_non_blue_scores_zero_and_flags_reason(name, hsv):
    result = _service().check(_swatch(*hsv))
    assert result.blue_coverage == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)
    assert result.reason == "no_blue_region"


def test_marketing_backdrop_blue_is_inside_the_band_by_design():
    """The saturation floor sits *below* backdrop blue deliberately.

    Real fabric loses saturation fast in poor light, so the floor is set to
    admit it. What separates a featureless blue field from a real uniform is the
    logo term scoring 0.0, capping the fused score at blue_weight — not a colour
    rejection. See limitation #1 in the design doc.
    """
    result = _service().check(_swatch(115, 167, 216))
    assert result.blue_coverage == pytest.approx(1.0)
    assert result.logo_match == pytest.approx(0.0)
    assert result.score == pytest.approx(0.7, abs=1e-6)


# -- ROI geometry ------------------------------------------------------------


def test_torso_rect_from_face_matches_documented_geometry():
    # Face box spanning x 0.4-0.6, y 0.1-0.3 in a 1000x1000 frame:
    # fw = fh = 0.2, cx = 0.5 -> x 0.18-0.82, y 0.36-0.90
    rect = DresscodeDetectionService._torso_rect_from_face([0.4, 0.1, 0.6, 0.3], 1000, 1000)
    assert rect == (180, 360, 820, 900)


def test_torso_rect_clips_to_frame_edges():
    """A face at the frame edge yields a clipped, still-valid rect."""
    rect = DresscodeDetectionService._torso_rect_from_face([0.0, 0.7, 0.3, 1.0], 100, 100)
    x1, y1, x2, y2 = rect
    assert 0 <= x1 <= x2 <= 100
    assert 0 <= y1 <= y2 <= 100


def test_no_face_uses_fallback_region():
    result = _service(bbox=None).check(_swatch(113, 241, 169))
    assert result.face_detected is False
    assert result.region == "fallback_full_image"


def test_face_detected_uses_torso_region():
    result = _service(bbox=[0.4, 0.05, 0.6, 0.25]).check(_swatch(113, 241, 169))
    assert result.face_detected is True
    assert result.region == "torso"


def test_roi_below_minimum_is_not_scored():
    tiny = _swatch(113, 241, 169, size=(20, 20))
    result = _service(min_roi_pixels=2000).check(tiny)
    assert result.reason == "roi_too_small"
    assert result.score == pytest.approx(0.0)
    assert result.logo_match is None


# -- fusion ------------------------------------------------------------------


def test_missing_logo_renormalizes_blue_to_full_weight():
    """A skipped logo term must not silently push a good image under threshold."""
    svc = _service(logo_enabled=False)
    result = svc.check(_swatch(113, 241, 169))
    assert result.logo_match is None
    # Blue coverage is 1.0; with the logo unavailable the score is blue alone,
    # not blue * blue_weight.
    assert result.score == pytest.approx(1.0)


def test_logo_term_skipped_when_chest_band_too_narrow():
    narrow = _swatch(113, 241, 169, size=(400, 40))
    result = _service(logo_min_chest_px=80, min_roi_pixels=100).check(narrow)
    assert result.logo_match is None


def test_fusion_weights_are_applied():
    svc = _service(blue_weight=0.5, logo_weight=0.5)
    # blue = 1.0, logo = 0.0 on a featureless swatch -> 0.5
    assert svc.check(_swatch(113, 241, 169)).score == pytest.approx(0.5, abs=1e-6)


def test_missing_template_disables_logo_term(tmp_path):
    svc = _service(logo_template_path=tmp_path / "does_not_exist.png")
    assert svc.logo_available is False
    assert svc.check(_swatch(113, 241, 169)).logo_match is None


# -- learned classifier ------------------------------------------------------


class _FakeClassifier:
    threshold = 0.35

    def __init__(self, prob: float) -> None:
        self._prob = prob
        self.calls: list = []

    def score_all(self, image_rgb, face_bbox):
        self.calls.append(face_bbox)
        return {"uniform": self._prob, "dress_color": 0.8, "logo": 0.1}

    def thresholds(self):
        return {"uniform": self.threshold, "dress_color": 0.38, "logo": 0.4}


def test_classifier_score_decides_and_colour_is_reported_alongside():
    clf = _FakeClassifier(0.91)
    red = _swatch(0, 240, 200)  # no blue at all: the rule would flag no_blue_region
    result = _service([0.4, 0.1, 0.6, 0.3], classifier=clf).check(red)
    assert result.engine == "siglip2"
    assert result.score == pytest.approx(0.91)
    assert result.uniform_score == pytest.approx(0.91)
    assert result.recommended_threshold == pytest.approx(0.35)
    assert result.reason is None                 # the rule's veto does not apply
    assert result.blue_coverage == pytest.approx(0.0)
    assert clf.calls == [[0.4, 0.1, 0.6, 0.3]]    # torso crop anchored on the face box
    assert result.check_scores == {"dress_color": 0.8, "logo": 0.1}
    assert result.check_thresholds == {"dress_color": 0.38, "logo": 0.4}


def test_classifier_gets_no_box_when_no_face():
    clf = _FakeClassifier(0.2)
    _service(None, classifier=clf).check(_swatch(113, 241, 169))
    assert clf.calls == [None]


def test_classifier_still_scores_when_colour_roi_is_too_small():
    result = _service(min_roi_pixels=10**9, classifier=_FakeClassifier(0.7)).check(_swatch(113, 241, 169))
    assert result.score == pytest.approx(0.7)
    assert result.reason is None
    assert result.logo_match is None


def test_without_classifier_engine_is_hsv():
    result = _service([0.4, 0.1, 0.6, 0.3]).check(_swatch(113, 241, 169))
    assert result.engine == "hsv"
    assert result.uniform_score is None
