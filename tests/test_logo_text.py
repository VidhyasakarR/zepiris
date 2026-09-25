"""Logo check by OCR: LOADSHARE letters (partial runs ok, any direction) on Loadshare blue."""

from __future__ import annotations

import numpy as np
import pytest

from zepiris.ml_inference.logo_text import MIN_SIMILARITY, LoadshareTextDetector, similarity

BLUE_RGB = (30, 60, 220)     # Loadshare blue: OpenCV hue ~115
BLACK_RGB = (20, 20, 20)


@pytest.mark.parametrize("text", [
    "LOADSHARE", "loadshare", "SHARE", "LOAD", "ADSHARE", "OADSH", "HARE",
    "ERAHSDAOL",      # read bottom-to-top
    "ERAHS",          # partial, reversed
    "LOADSHARF",      # one OCR misread on a long read
    "LLOADSHARE",     # doubled edge letter
    "LLOAD",
    "LOAD SHARE",     # split by a space
    "HARF", "DSHARF", # a cut E reads as F; LOADSHARE has no F
    "LOADSHARC",      # stray letter at the end of a long read
])
def test_accepted_reads(text: str) -> None:
    assert similarity(text)[0] >= MIN_SIMILARITY, text


@pytest.mark.parametrize("text", [
    "SWIGGY", "ZOMATO", "ZEPTO", "BLINKIT", "AMAZON", "MARATHON", "RAPIDO", "FLIPKART",
    "ROADS",          # partly-read ROADSTAR: short reads must be exact
    "ADSTAR", "SHAREIT", "ROADSTAR",
    "LOADSTAR", "OADSTAR",  # a wrong letter mid-word is another word
    "ARE",            # 3 letters only count when cut off by the photo edge
    "LOA", "SHA",     # under 4 letters
    "FOVDEHVE",       # mirrored wordmark: handled by re-reading, not by the matcher
    "",
])
def test_rejected_reads(text: str) -> None:
    assert similarity(text)[0] < MIN_SIMILARITY, text


class _FakeOCR:
    """Returns scripted readings, one list per call (as-is, mirrored, rotated)."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self, _img_bgr):
        self.calls += 1
        r = self.readings.pop(0) if self.readings else []
        return r, None


def _shirt(colour=BLUE_RGB, h=400, w=300) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = colour
    img[150:250, 120:160] = 250          # white ink where the word is
    return img


BOX = [[110, 140], [170, 140], [170, 260], [110, 260]]


def test_word_on_blue_passes() -> None:
    d = LoadshareTextDetector(ocr=_FakeOCR([(BOX, "LOADSHARE", 0.97)]))
    r = d.detect(_shirt(), None)
    assert r.score >= MIN_SIMILARITY and r.matched == "LOADSHARE" and r.reason is None
    assert r.on_blue > 0.9


def test_word_on_black_shirt_fails_without_reading() -> None:
    ocr = _FakeOCR([(BOX, "LOADSHARE", 0.97)])
    r = LoadshareTextDetector(ocr=ocr).detect(_shirt(BLACK_RGB), None)
    assert r.score == 0 and r.reason == "not_on_blue"
    assert ocr.calls == 0            # no blue fabric: rejected before OCR


def test_other_word_on_blue_fails() -> None:
    r = LoadshareTextDetector(ocr=_FakeOCR([(BOX, "SWIGGY", 0.99)], [], [])).detect(_shirt(), None)
    assert r.score == 0 and r.reason == "no_loadshare_text"


def test_mirrored_selfie_is_read_again() -> None:
    ocr = _FakeOCR([(BOX, "FOVDEHVE", 0.7)], [(BOX, "LOADSHARE", 0.99)])
    r = LoadshareTextDetector(ocr=ocr).detect(_shirt(), None)
    assert r.score >= MIN_SIMILARITY and ocr.calls == 2


def test_upside_down_is_read_again() -> None:
    ocr = _FakeOCR([(BOX, "HRV", 0.9)], [], [(BOX, "LOADSHARF", 0.9)])
    r = LoadshareTextDetector(ocr=ocr).detect(_shirt(), None)
    assert r.score >= MIN_SIMILARITY and ocr.calls == 3


def test_text_on_a_white_patch_is_not_on_blue() -> None:
    img = _shirt()
    img[100:300, 80:220] = 245       # a white poster / patch around the word
    r = LoadshareTextDetector(ocr=_FakeOCR([(BOX, "LOADSHARE", 0.97)], [], [])).detect(img, None)
    assert r.score == 0 and r.reason == "not_on_blue"


def test_shirt_region_is_below_the_face() -> None:
    img = np.zeros((1000, 600, 3), np.uint8)
    region = LoadshareTextDetector.shirt_region(img, [0.4, 0.1, 0.6, 0.25])
    # starts under the chin (y1 + 0.9 face heights), not at the top of the frame
    assert region.shape[0] < 1000 and region.shape[0] > 500


@pytest.mark.parametrize("text,ok", [
    ("ARE", True), ("LOA", True), ("ERA", True),   # 3 letters of a cut-off wordmark
    ("IARE", True), ("OHARF", True),              # the sliced letter misread
    ("RE", False), ("THE", False), ("STAR", False), ("ADSTAR", False), ("SWIG", False),
])
def test_cut_off_reads(text: str, ok: bool) -> None:
    assert (similarity(text, cut=True)[0] >= MIN_SIMILARITY) is ok, text


def _chest_only(h=300, w=300) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = BLUE_RGB
    img[200:300, 120:150] = 250
    return img


VERTICAL_AT_BOTTOM = [[118, 200], [152, 200], [152, 299], [118, 299]]   # runs into the frame edge
VERTICAL_MID = [[118, 100], [152, 100], [152, 200], [118, 200]]


def test_cut_off_wordmark_at_frame_edge_passes() -> None:
    r = LoadshareTextDetector(ocr=_FakeOCR([(VERTICAL_AT_BOTTOM, "ARE", 0.9)])).detect(_chest_only(), None)
    assert r.score >= MIN_SIMILARITY and r.matched == "ARE"


def test_three_letters_mid_shirt_do_not_pass() -> None:
    r = LoadshareTextDetector(ocr=_FakeOCR([(VERTICAL_MID, "ARE", 0.9)], [], [])).detect(_chest_only(), None)
    assert r.score == 0
