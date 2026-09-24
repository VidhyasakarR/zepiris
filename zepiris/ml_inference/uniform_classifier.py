"""Learned uniform classifier: SigLIP2 image embeddings + a trained linear head.

Why a learned model on top of the colour/logo check: blue coverage cannot tell a
Loadshare polo from any other blue shirt, and template matching the small logo
is fragile. Trained on 18 riders (two shirt designs), this classifier accepted
98% of held-out real uniforms and rejected 100% of the same shirts with the
print painted out, where the colour/logo rule accepted 63% of those.

Two views of the image are embedded in one batch — the whole frame and a
face-anchored chest-to-stomach crop — and the head scores their concatenation.
Both views, the crop geometry and the preprocessing must match training exactly
(see scripts/export_dresscode_model.py and the head's own metadata).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_HEAD_PATH = Path(__file__).parent / "assets" / "dresscode_head.json"
_INPUT = 224


class UniformClassifier:
    """Probability that the subject is wearing the Loadshare uniform.

    Args:
        encoder_path: SigLIP2 vision encoder exported to ONNX.
        head_path: Trained head JSON (coefficients, intercept, crop geometry,
            recommended threshold).
        providers: onnxruntime execution providers, CUDA first when available.
    """

    def __init__(
        self,
        encoder_path: str | Path,
        head_path: str | Path = DEFAULT_HEAD_PATH,
        providers: list[str] | None = None,
    ) -> None:
        head = json.loads(Path(head_path).read_text())
        self._coef = np.asarray(head["coef"], dtype=np.float32)
        self._intercept = float(head["intercept"])
        crop = head["torso_crop_faces"]
        self._half_w, self._top, self._bottom = crop["half_width"], crop["top"], crop["bottom"]
        self.threshold = float(head["threshold"])
        self.model_name = head.get("model", "siglip2")
        self._session = ort.InferenceSession(
            str(encoder_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        dim = self._session.get_outputs()[0].shape[-1]
        if isinstance(dim, int) and dim * 2 != self._coef.size:
            raise ValueError(f"head expects {self._coef.size // 2}-d embeddings, encoder gives {dim}")

    def torso_crop(self, image_rgb: np.ndarray, bbox: list[float] | None) -> np.ndarray:
        """Chest-to-stomach crop from a normalized face box; the whole frame when there is none."""
        h, w = image_rgb.shape[:2]
        if not bbox:
            return image_rgb
        x1, y1, x2, y2 = bbox
        fw, fh, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
        X1, X2 = max(0.0, cx - self._half_w * fw), min(1.0, cx + self._half_w * fw)
        Y1, Y2 = max(0.0, y1 + self._top * fh), min(1.0, y1 + self._bottom * fh)
        crop = image_rgb[int(Y1 * h):int(Y2 * h), int(X1 * w):int(X2 * w)]
        return crop if crop.size and min(crop.shape[:2]) >= 16 else image_rgb

    @staticmethod
    def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
        # PIL bilinear, not cv2: it is what the training embeddings were made
        # with. cv2.INTER_LINEAR does not antialias on downscale and moved the
        # embeddings to cosine ~0.91 of the trained ones — enough to skew scores.
        resized = Image.fromarray(image_rgb).resize((_INPUT, _INPUT), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        return ((arr / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def score(self, image_rgb: np.ndarray, face_bbox: list[float] | None) -> float:
        """Uniform probability in [0, 1] for one RGB image."""
        views = np.stack([
            self._preprocess(image_rgb),
            self._preprocess(self.torso_crop(image_rgb, face_bbox)),
        ])
        emb = self._session.run(None, {self._input: views})[0].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        logit = float(emb.reshape(-1) @ self._coef + self._intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))
