"""Dress-code detection: is the subject wearing the blue Loadshare uniform shirt?

Two independent signals are combined:

* **Blue coverage** — how much of the torso falls inside the Loadshare-blue HSV
  band. Calibrated from reference photos (shirt fabric sits at hue ~113 with
  saturation ~241). Cheap and robust, but colour alone cannot tell one brand's
  blue shirt from another's.
* **Logo match** — normalized template correlation against the chest logo. This
  is the only term that actually discriminates *Loadshare* from *blue*, and it
  is resolution-bound: the logo is small, so below a usable chest width the term
  is skipped rather than guessed at.

No model weights are loaded here. The torso is located from the face box the
already-loaded InsightFace detector produces, so this adds a colour pass and a
template sweep to an image the service was going to detect on anyway.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from zepiris.schemas.ml_inference import DresscodeCheckResult

logger = logging.getLogger(__name__)

DEFAULT_LOGO_PATH = Path(__file__).parent / "assets" / "loadshare_logo.png"

# Torso ROI geometry, expressed in multiples of the detected face box. The
# shirt sits below the chin and spans wider than the head; these multiples put
# the ROI over the chest without reaching the background on a typical selfie.
_ROI_HALF_WIDTH_FACES = 1.6
_ROI_TOP_FACES = 1.3
_ROI_BOTTOM_FACES = 4.0

# Fallback ROI when no face anchors the torso: central 60% horizontally of the
# lower 70% of the frame. Materially weaker than a face-anchored ROI — with no
# anchor, a blue background lands directly inside it — so the caller is told
# via ``region`` that the result came from here.
_FALLBACK_X1, _FALLBACK_X2 = 0.20, 0.80
_FALLBACK_Y1, _FALLBACK_Y2 = 0.30, 1.00

#: Below this blue coverage the ROI has effectively no blue in it at all.
_NO_BLUE_FLOOR = 0.02

#: Template widths to sweep, as a fraction of the ROI width.
_LOGO_SCALES = (0.12, 0.17, 0.22, 0.28, 0.34, 0.42)

# White-ish pixels: the logo is white on blue, so low saturation + high value.
_LOGO_WHITE_SAT_MAX = 70
_LOGO_WHITE_VAL_MIN = 150


class DresscodeDetectionService:
    """Score whether a photo shows the blue Loadshare uniform shirt."""

    def __init__(
        self,
        face_service,
        *,
        hue_min: int = 100,
        hue_max: int = 122,
        sat_min: int = 110,
        val_min: int = 50,
        val_max: int = 245,
        blue_weight: float = 0.7,
        logo_weight: float = 0.3,
        logo_enabled: bool = True,
        logo_min_chest_px: int = 80,
        min_roi_pixels: int = 2000,
        logo_template_path: str | Path | None = None,
        classifier=None,
    ) -> None:
        """
        Args:
            face_service: Loaded ``FaceEmbeddingService``; only ``detect_box`` is used.
            hue_min/hue_max: Loadshare-blue hue band, OpenCV scale (0-179).
            sat_min: Saturation floor. Deliberately set below the observed
                marketing-backdrop value rather than above it — real fabric
                loses saturation fast in poor light, and rejecting a genuinely
                uniformed rider is worse than scoring a blue backdrop. The logo
                term, not this floor, is the brand discriminator.
            val_min/val_max: Brightness band; excludes near-black folds and blown highlights.
            blue_weight/logo_weight: Fusion weights. Renormalized when the logo is skipped.
            logo_enabled: Disables the template sweep entirely.
            logo_min_chest_px: Below this chest width the logo term is skipped (reported null).
            min_roi_pixels: ROIs smaller than this are not scored.
            logo_template_path: Override for the committed logo asset.
            classifier: Optional :class:`~zepiris.ml_inference.uniform_classifier.UniformClassifier`.
                When set, its probability is the decision score (``engine="siglip2"``)
                and blue coverage / logo match are reported as supporting signals
                only. Without it the fused colour/logo score decides (``engine="hsv"``).
        """
        self._face_service = face_service
        self._hue_min = int(hue_min)
        self._hue_max = int(hue_max)
        self._sat_min = int(sat_min)
        self._val_min = int(val_min)
        self._val_max = int(val_max)
        self._blue_weight = float(blue_weight)
        self._logo_weight = float(logo_weight)
        self._logo_enabled = bool(logo_enabled)
        self._logo_min_chest_px = int(logo_min_chest_px)
        self._min_roi_pixels = int(min_roi_pixels)
        self._templates = self._load_templates(logo_template_path or DEFAULT_LOGO_PATH)
        self._classifier = classifier

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def _load_templates(path: str | Path) -> list[np.ndarray]:
        """Load the logo template as grayscale, plus its mirror.

        Front-facing phone cameras mirror the frame, so roughly half of real
        selfies show the logo reversed. Both orientations are matched and the
        better score wins; without the mirror, every mirrored selfie would score
        near zero on an otherwise perfectly visible logo.
        """
        p = Path(path)
        if not p.is_file():
            logger.warning("Dresscode logo template not found at %s; logo term disabled", p)
            return []
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Dresscode logo template at %s could not be decoded", p)
            return []
        return [img, cv2.flip(img, 1)]

    @property
    def logo_available(self) -> bool:
        return self._logo_enabled and bool(self._templates)

    # -- geometry ------------------------------------------------------------

    @staticmethod
    def _torso_rect_from_face(
        bbox: list[float], width: int, height: int
    ) -> tuple[int, int, int, int]:
        """Map a normalized face box to a pixel torso rect, clipped to the frame."""
        x1, y1, x2, y2 = bbox
        fw = max(1e-6, x2 - x1)
        fh = max(1e-6, y2 - y1)
        cx = (x1 + x2) / 2.0

        rx1 = cx - _ROI_HALF_WIDTH_FACES * fw
        rx2 = cx + _ROI_HALF_WIDTH_FACES * fw
        ry1 = y1 + _ROI_TOP_FACES * fh
        ry2 = y1 + _ROI_BOTTOM_FACES * fh
        return _to_pixels(rx1, ry1, rx2, ry2, width, height)

    @staticmethod
    def _fallback_rect(width: int, height: int) -> tuple[int, int, int, int]:
        return _to_pixels(
            _FALLBACK_X1, _FALLBACK_Y1, _FALLBACK_X2, _FALLBACK_Y2, width, height
        )

    # -- scoring -------------------------------------------------------------

    def _blue_coverage(self, roi_rgb: np.ndarray) -> float:
        """Fraction of the ROI inside the Loadshare-blue HSV band."""
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([self._hue_min, self._sat_min, self._val_min], dtype=np.uint8)
        upper = np.array([self._hue_max, 255, self._val_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _logo_score(self, roi_rgb: np.ndarray) -> float | None:
        """Best normalized template correlation for the logo in the chest band.

        Returns ``None`` when the term could not be evaluated (no template, or a
        chest band too small for the logo to survive), which is reported to the
        caller rather than folded in as a zero.
        """
        if not self.logo_available:
            return None

        h, w = roi_rgb.shape[:2]
        if w < self._logo_min_chest_px:
            return None

        # The logo sits on the upper chest, so only the top half of the torso
        # ROI is searched — halving the sweep area and removing false peaks from
        # folds and hems lower down.
        band = roi_rgb[: max(1, h // 2), :]
        bh, bw = band.shape[:2]
        if bh < 8 or bw < 8:
            return None

        gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)

        # A chest band with no white-ish pixels cannot hold a white logo; skip
        # the (comparatively expensive) multi-scale sweep entirely.
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
        white = cv2.inRange(
            hsv,
            np.array([0, 0, _LOGO_WHITE_VAL_MIN], dtype=np.uint8),
            np.array([179, _LOGO_WHITE_SAT_MAX, 255], dtype=np.uint8),
        )
        if np.count_nonzero(white) < 20:
            return 0.0

        best = 0.0
        evaluated = False
        for template in self._templates:
            th0, tw0 = template.shape[:2]
            for scale in _LOGO_SCALES:
                tw = max(8, int(round(bw * scale)))
                th = max(8, int(round(tw * th0 / tw0)))
                if th > bh or tw > bw:
                    continue
                resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                evaluated = True
                best = max(best, float(result.max()))

        if not evaluated:
            return None
        return max(0.0, min(1.0, best))

    def _fuse(self, blue: float, logo: float | None) -> float:
        """Weighted fusion of the two terms.

        When the logo term is unavailable its weight is not simply dropped — the
        blue term is renormalized to carry the full weight. Otherwise a
        low-resolution image would be pushed under the threshold by a missing
        signal rather than by anything observed in it.
        """
        if logo is None:
            return max(0.0, min(1.0, blue))
        total = self._blue_weight + self._logo_weight
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, (self._blue_weight * blue + self._logo_weight * logo) / total))

    # -- entry point ---------------------------------------------------------

    def check(self, image_rgb: np.ndarray) -> DresscodeCheckResult:
        """Score one image. Never raises for a missing face — that is a fallback, not an error.

        Args:
            image_rgb: Image in RGB format, shape (H, W, 3), dtype uint8.

        Returns:
            DresscodeCheckResult: sub-scores, fused score and provenance. The
            decision threshold is applied by the caller, not here.
        """
        height, width = image_rgb.shape[:2]

        detection = self._face_service.detect_box(image_rgb)
        if detection.face_detected:
            rect = self._torso_rect_from_face(detection.bbox, width, height)
            region = "torso"
        else:
            rect = self._fallback_rect(width, height)
            region = "fallback_full_image"

        x1, y1, x2, y2 = rect
        roi = image_rgb[y1:y2, x1:x2]
        roi_ok = roi.size > 0 and roi.shape[0] * roi.shape[1] >= self._min_roi_pixels

        if self._classifier is not None:
            # The learned model scores the whole frame plus its own torso crop,
            # so a small colour ROI does not stop it; the colour/logo terms are
            # still computed where possible and reported alongside.
            probs = self._classifier.score_all(
                image_rgb, detection.bbox if detection.face_detected else None
            )
            uniform = probs["uniform"]
            return DresscodeCheckResult(
                face_detected=detection.face_detected,
                region=region,
                blue_coverage=self._blue_coverage(roi) if roi_ok else 0.0,
                logo_match=self._logo_score(roi) if roi_ok else None,
                score=uniform,
                uniform_score=uniform,
                engine="siglip2",
                recommended_threshold=self._classifier.threshold,
                check_scores={k: v for k, v in probs.items() if k != "uniform"},
                check_thresholds={
                    k: v for k, v in self._classifier.thresholds().items() if k != "uniform"
                },
                reason=None,
            )

        if not roi_ok:
            return DresscodeCheckResult(
                face_detected=detection.face_detected,
                region=region,
                blue_coverage=0.0,
                logo_match=None,
                score=0.0,
                reason="roi_too_small",
            )

        blue = self._blue_coverage(roi)
        logo = self._logo_score(roi)
        score = self._fuse(blue, logo)
        reason = "no_blue_region" if blue < _NO_BLUE_FLOOR else None

        return DresscodeCheckResult(
            face_detected=detection.face_detected,
            region=region,
            blue_coverage=blue,
            logo_match=logo,
            score=score,
            reason=reason,
        )


def _to_pixels(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a normalized rect to [0, 1] and convert it to pixel bounds."""
    px1 = int(round(max(0.0, min(1.0, x1)) * width))
    px2 = int(round(max(0.0, min(1.0, x2)) * width))
    py1 = int(round(max(0.0, min(1.0, y1)) * height))
    py2 = int(round(max(0.0, min(1.0, y2)) * height))
    return px1, py1, max(px1, px2), max(py1, py2)
