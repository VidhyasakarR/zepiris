"""UniformClassifier against the real exported encoder (skipped when absent).

Locks in the two things that silently wreck scores if they drift: the head's
shape matching the encoder, and preprocessing that reproduces training.
"""

from pathlib import Path

import numpy as np
import pytest

from zepiris.ml_inference.uniform_classifier import UniformClassifier

_ENCODER = Path("models/siglip2_base_vision.onnx")
pytestmark = pytest.mark.skipif(not _ENCODER.exists(), reason="run scripts/export_dresscode_model.py")


@pytest.fixture(scope="module")
def clf() -> UniformClassifier:
    return UniformClassifier(_ENCODER)


def test_score_is_a_probability(clf) -> None:
    img = np.full((400, 300, 3), (30, 60, 200), dtype=np.uint8)
    p = clf.score(img, [0.35, 0.05, 0.65, 0.3])
    assert 0.0 <= p <= 1.0


def test_plain_grey_frame_is_not_a_uniform(clf) -> None:
    img = np.full((400, 300, 3), 128, dtype=np.uint8)
    assert clf.score(img, None) < clf.threshold


def test_torso_crop_geometry(clf) -> None:
    img = np.zeros((1000, 500, 3), dtype=np.uint8)
    crop = clf.torso_crop(img, [0.4, 0.1, 0.6, 0.2])  # fw=0.2, fh=0.1, cx=0.5
    # x: 0.5 +/- 2.2*0.2 -> [0.06, 0.94]; y: 0.1+0.1 .. 0.1+0.55 -> [0.2, 0.65]
    assert crop.shape[:2] == (450, 440)
