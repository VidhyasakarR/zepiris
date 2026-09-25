"""Capture quality: is the face — and the T-shirt — clearly visible in this selfie?

Runs right after the rider captures, before the checkpoint, so a photo that
would fail face match or the dress checks can be retaken on the spot.

* **Face**: found at all, big enough, bright enough, and sharp — the ResNet-18
  blur model on the face crop. On a set of rider photos degraded with focus,
  motion and defocus blur (half of them also dark + sensor noise, the case the
  old in-browser metric misread as sharp), it caught 98% of blurred faces and
  flagged 8% of sharp ones (AUC 0.994), and 94% / 7% in low light. It also
  flagged both real blurry captures that the browser metric had passed.
* **T-shirt**: the chest-to-stomach region below the face is inside the frame,
  and not blurry. Plain fabric has little texture, which the blur model reads
  as soft, so the shirt uses a stricter bar (0.95: 75% of blur caught, 2% of
  sharp shirts flagged) than the face (0.5).
"""

from __future__ import annotations

import numpy as np

from zepiris.schemas.ml_inference import CaptureQualityResult

#: Blur-model probability above which the region is "blurry".
FACE_BLUR_THRESHOLD = 0.5
SHIRT_BLUR_THRESHOLD = 0.95
#: Face narrower than this fraction of the frame is too far away to verify.
MIN_FACE_WIDTH_FRAC = 0.12
#: Mean luma (0–255) of the face under which there is too little light, and
#: over which it is overexposed (skin detail clipped, face match degrades).
TOO_DARK_LUMA = 45
TOO_BRIGHT_LUMA = 230
#: The expected T-shirt region (chin → stomach) must be at least this much in frame.
#: 0.35: a close selfie that shows the chest and upper stomach is enough for the
#: colour and logo checks; 0.5 of a region reaching 4 face-heights down flagged
#: exactly such a photo ("T-shirt not visible" with the whole chest in view).
MIN_SHIRT_VISIBLE = 0.35

# T-shirt region, in face-box units: from just under the chin to the stomach
# (about three face-heights below the top of the head).
_SHIRT_HALF_W, _SHIRT_TOP, _SHIRT_BOTTOM = 2.0, 0.3, 3.0


def _crop(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = img.shape[:2]
    X1, X2 = int(max(0.0, x1) * w), int(min(1.0, x2) * w)
    Y1, Y2 = int(max(0.0, y1) * h), int(min(1.0, y2) * h)
    return img[Y1:Y2, X1:X2]


class CaptureQualityService:
    """Grades face and T-shirt visibility / clarity of one RGB selfie.

    Args:
        face_service: provides ``detect_box`` (normalized face box).
        blur_service: the ResNet-18 blur model (``BlurDetectionService``).
    """

    def __init__(self, face_service, blur_service) -> None:
        self._face = face_service
        self._blur = blur_service

    def _blur_prob(self, crop: np.ndarray) -> float:
        return float(self._blur.predict(self._blur.preprocess(crop))[0])

    def check(self, image_rgb: np.ndarray) -> CaptureQualityResult:
        det = self._face.detect_box(image_rgb)
        if not det.face_detected:
            return CaptureQualityResult(face_detected=False, issues=["face_not_found"])

        x1, y1, x2, y2 = det.bbox
        fw, fh, cx, cy = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        issues: list[str] = []

        # -- face: size, light, sharpness (square crop around the face, 1.2× padded)
        h, w = image_rgb.shape[:2]
        side = max(fw * w, fh * h) * 1.2
        face = _crop(image_rgb, cx - side / 2 / w, cy - side / 2 / h, cx + side / 2 / w, cy + side / 2 / h)
        brightness = float((face.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
        face_blur = self._blur_prob(face) if min(face.shape[:2]) >= 16 else 1.0
        if fw < MIN_FACE_WIDTH_FRAC:
            issues.append("face_too_small")
        if brightness < TOO_DARK_LUMA:
            issues.append("too_dark")
        elif brightness > TOO_BRIGHT_LUMA:
            issues.append("too_bright")
        elif face_blur > FACE_BLUR_THRESHOLD:
            # in the dark "too dark" is the actionable message; blur follows from it
            issues.append("face_blurry")

        # -- T-shirt: the chin-to-stomach region below the face
        top, bottom = y2 + _SHIRT_TOP * fh, y1 + _SHIRT_BOTTOM * fh
        visible = max(0.0, min(1.0, bottom) - min(1.0, top)) / max(1e-6, bottom - top)
        shirt = _crop(image_rgb, cx - _SHIRT_HALF_W * fw, top, cx + _SHIRT_HALF_W * fw, bottom)
        shirt_blur = None
        if visible < MIN_SHIRT_VISIBLE or shirt.size == 0 or min(shirt.shape[:2]) < 32:
            issues.append("shirt_not_visible")
        else:
            shirt_blur = self._blur_prob(shirt)
            if shirt_blur > SHIRT_BLUR_THRESHOLD and "too_dark" not in issues:
                issues.append("shirt_blurry")

        return CaptureQualityResult(
            face_detected=True,
            face_width=round(fw, 3),
            face_brightness=round(brightness, 1),
            face_blur=round(face_blur, 3),
            shirt_visible=round(visible, 3),
            shirt_blur=None if shirt_blur is None else round(shirt_blur, 3),
            issues=issues,
        )
