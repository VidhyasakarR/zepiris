"""Logo check by reading the shirt: the letters of LOADSHARE, printed on Loadshare blue.

The learned logo head (a SigLIP2 probe) had only ever seen "Loadshare print"
against "no print", so it learned *white print on the chest*: a marathon tee
passed. Reading the text answers the actual question.

Rule:
* OCR (PP-OCR via RapidOCR, ONNX, CPU) reads every text line in the T-shirt
  region below the face, in any orientation. The big wordmark runs down the
  front; lines read top-to-bottom or bottom-to-top both count, and so do
  horizontal ones such as the back print or the chest logo's "LOADSHARE".
* A line counts when its letters are a run of LOADSHARE ("LOADSHARE", "SHARE",
  "ADSHA", "LOAD", ...) of at least 4 letters, forwards or backwards, allowing
  OCR misreads, but only on longer reads and only at the ends: up to 6 letters
  must match a run exactly ("SHARE", "LOAD", "ADSHAR"); 7+ letters may have one
  wrong or extra letter at the first or last position ("LOADSHARC"). A wrong
  letter mid-word is another word ("ROADSTAR", "LOADSTAR", "OADSTAR").
* And it must sit on Loadshare blue: at least MIN_BLUE of the fabric around the
  letters (white ink excluded) is Loadshare blue, so blue text on a white poster,
  or a black tee with the word on it, does not count.

A chest-only photo slices the vertical wordmark at the frame edge. Vertical text
that runs into the photo's top or bottom edge therefore needs only 3 letters
("ARE", "LOA") and may have one wrong letter, the sliced one ("IARE"). A cut E
reads as F, and LOADSHARE has no F, so F is always read as E.

If nothing matches, the region is read again mirrored (a selfie saved mirrored
by the phone's camera app reads "FOVDEHVE") and upside down. A region with
almost no Loadshare-blue fabric is rejected without reading at all: a logo only
counts on blue, and it keeps non-uniform photos fast.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET = "LOADSHARE"
MIN_RUN = 4
MIN_SIMILARITY = 0.8
FUZZY_FROM = 7   # reads this long may have one OCR error
CUT_MIN_RUN = 3  # a vertical wordmark cut off by the photo edge: 3 letters do
_EDGE = 0.03     # "touches the edge": within 3% of the photo border
MIN_BLUE = 0.45
MAX_SIDE = 1280
MIN_REGION_BLUE = 0.08   # below this share of blue fabric, no OCR: it cannot pass

# Loadshare-blue fabric in OpenCV HSV (H 0-179): the same band the shipped
# dress-colour data was built with.
_HUE = (95, 130)
_SAT_MIN, _VAL_MIN = 80, 40

# T-shirt region in face-box units, generous: the wordmark runs from the
# collar down past the stomach, and the chest logo sits off to one side.
_HALF_W, _TOP, _BOTTOM = 2.6, 0.9, 7.0


_RUNS = [TARGET[i:j] for i in range(len(TARGET)) for j in range(i + CUT_MIN_RUN, len(TARGET) + 1)]

# OCR look-alikes, mapped before matching. LOADSHARE has no F: an E whose bottom
# arm is sliced off by the frame edge reads as F ("HARF", "DSHARF").
_LOOKALIKE = str.maketrans({"F": "E", "0": "O", "5": "S"})


def letters(text: str) -> str:
    """Upper-case letters only (look-alikes mapped), doubled letters collapsed:
    LOADSHARE has no double letter, and OCR often doubles an edge letter ("LLOAD")."""
    t = text.upper().translate(_LOOKALIKE)
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^A-Z]", "", t))


def _end_error(cand: str, run: str) -> bool:
    """One wrong or extra letter, at the very start or end only.

    That is where OCR misreads land in practice: the letter the photo edge slices
    ("IARE", "HARF"), a stray edge letter ("LOADSHARC", "LOADSHAREI"). A wrong
    letter in the middle is a different word ("STAR", "ROADSTAR", "OADSTAR").
    """
    if len(cand) == len(run):
        diff = [i for i, (a, b) in enumerate(zip(cand, run)) if a != b]
        return len(diff) == 1 and diff[0] in (0, len(run) - 1)
    if len(cand) == len(run) + 1:
        return cand[1:] == run or cand[:-1] == run
    return False


def similarity(text: str, cut: bool = False) -> tuple[float, str]:
    """Best match of ``text`` against any run of LOADSHARE (either direction).

    * Plain read: 4+ letters, exact ("SHARE", "LOAD", "ADSHAR"); 7+ letters may
      have one wrong or extra letter at either end ("LOADSHARF").
    * ``cut`` (vertical text running off the photo edge, a chest-only selfie):
      3 letters are enough ("ARE", "LOA") and 4+ letters may have one wrong or
      extra letter at either end (the sliced one: "IARE" for HARE).

    Returns (similarity, matched run); 1.0 for an exact run, MIN_SIMILARITY for
    an end-letter match, 0 for no match.
    """
    tok = letters(text)
    min_run = CUT_MIN_RUN if cut else MIN_RUN
    if len(tok) < min_run:
        return 0.0, ""
    fuzzy = len(tok) >= (CUT_MIN_RUN + 1 if cut else FUZZY_FROM)
    best, run = 0.0, ""
    for cand in (tok, tok[::-1]):
        for r in _RUNS:
            if len(r) < min_run:
                continue
            if cand == r:
                return 1.0, r
            if fuzzy and best < MIN_SIMILARITY and _end_error(cand, r):
                best, run = MIN_SIMILARITY, r
    return best, run


@dataclass
class LogoTextResult:
    score: float                 # similarity of the best on-blue match; 0 if none
    text: str | None = None      # what was read there
    matched: str | None = None   # the LOADSHARE run it matched
    on_blue: float | None = None # blue share of the fabric around it
    reason: str | None = None    # "no_loadshare_text" | "not_on_blue" | None


class LoadshareTextDetector:
    """Reads the T-shirt and scores how clearly LOADSHARE is printed on blue fabric."""

    def __init__(self, ocr=None) -> None:
        if ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            ocr = RapidOCR()
        self._ocr = ocr

    # -- region ----------------------------------------------------------------
    @staticmethod
    def shirt_region(image_rgb: np.ndarray, face_bbox: list[float] | None) -> np.ndarray:
        return LoadshareTextDetector._region(image_rgb, face_bbox)[0]

    @staticmethod
    def _region(image_rgb: np.ndarray, face_bbox: list[float] | None) -> tuple[np.ndarray, dict]:
        """The shirt crop, and which of its sides are the photo's own edges."""
        whole = {"top": True, "bottom": True, "left": True, "right": True}
        if not face_bbox:
            return image_rgb, whole
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = face_bbox
        fw, fh, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
        X1, X2 = max(0.0, cx - _HALF_W * fw), min(1.0, cx + _HALF_W * fw)
        Y1, Y2 = max(0.0, y1 + _TOP * fh), min(1.0, y1 + _BOTTOM * fh)
        crop = image_rgb[int(Y1 * h):int(Y2 * h), int(X1 * w):int(X2 * w)]
        if not crop.size or min(crop.shape[:2]) < 32:
            return image_rgb, whole
        return crop, {"top": Y1 <= 0.0, "bottom": Y2 >= 1.0, "left": X1 <= 0.0, "right": X2 >= 1.0}

    @staticmethod
    def _is_cut(box: np.ndarray, shape: tuple[int, int], edges: dict) -> bool:
        """Vertical text that runs into the photo's top or bottom edge."""
        h, w = shape
        xs, ys = box[:, 0], box[:, 1]
        vertical = (ys.max() - ys.min()) >= 1.5 * max(1.0, xs.max() - xs.min())
        at_bottom = edges["bottom"] and ys.max() >= h * (1 - _EDGE)
        at_top = edges["top"] and ys.min() <= h * _EDGE
        return bool(vertical and (at_bottom or at_top))

    # -- colour ----------------------------------------------------------------
    @staticmethod
    def blue_around(region_rgb: np.ndarray, box: np.ndarray) -> float:
        """Share of Loadshare-blue among the non-ink pixels in and around a text box."""
        hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(region_rgb.shape[:2], np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
        side = max(3, int(min(cv2.minAreaRect(box.astype(np.float32))[1]) * 0.6))
        mask = cv2.dilate(mask, np.ones((side, side), np.uint8)) > 0
        hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        ink = (s < 90) & (v > 120)                     # white / light-blue letters
        fabric = mask & ~ink
        n = int(fabric.sum())
        if n < 20:
            return 0.0
        blue = fabric & (hch >= _HUE[0]) & (hch <= _HUE[1]) & (s >= _SAT_MIN) & (v >= _VAL_MIN)
        return float(blue.sum()) / n

    @staticmethod
    def blue_share(region_rgb: np.ndarray) -> float:
        hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        return float(((h >= _HUE[0]) & (h <= _HUE[1]) & (s >= _SAT_MIN) & (v >= _VAL_MIN)).mean())

    # -- reading ---------------------------------------------------------------
    def _read(self, region_rgb: np.ndarray, edges: dict) -> tuple[LogoTextResult, bool]:
        """Best match in one reading, and whether any word-like text was seen at all."""
        res, _ = self._ocr(cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR))
        best = LogoTextResult(score=0.0, reason="no_loadshare_text")
        saw_text = any(len(letters(t)) >= MIN_RUN for _b, t, _c in res or [])
        for box, text, _conf in res or []:
            pts = np.asarray(box, np.float32)
            sim, run = similarity(text, cut=self._is_cut(pts, region_rgb.shape[:2], edges))
            if sim < MIN_SIMILARITY:
                continue
            blue = self.blue_around(region_rgb, pts)
            score = sim if blue >= MIN_BLUE else 0.0
            cand = LogoTextResult(score=score, text=text, matched=run, on_blue=round(blue, 3),
                                  reason=None if score else "not_on_blue")
            # prefer an on-blue match, then the closer / longer one
            if (cand.score, len(run), sim) > (best.score, len(best.matched or ""), 0.0) or best.text is None:
                best = cand
        return best, saw_text

    def detect(self, image_rgb: np.ndarray, face_bbox: list[float] | None) -> LogoTextResult:
        region, edges = self._region(image_rgb, face_bbox)
        # OCR time grows with pixels; the wordmark is large, 1280 px is plenty.
        k = MAX_SIDE / max(region.shape[:2])
        if k < 1:
            region = cv2.resize(region, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if self.blue_share(region) < MIN_REGION_BLUE:
            return LogoTextResult(score=0.0, on_blue=0.0, reason="not_on_blue")
        best, _ = self._read(region, edges)
        mirrored = {**edges, "left": edges["right"], "right": edges["left"]}
        rotated = {"top": edges["bottom"], "bottom": edges["top"], "left": edges["right"], "right": edges["left"]}
        for variant, e in ((lambda r: r[:, ::-1].copy(), mirrored), (lambda r: cv2.rotate(r, cv2.ROTATE_180), rotated)):
            if best.score >= MIN_SIMILARITY:
                break
            again, _ = self._read(variant(region), e)
            if again.score > best.score or (best.text is None and again.text):
                best = again
        return best
