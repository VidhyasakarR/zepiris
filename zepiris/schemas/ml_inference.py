"""Pydantic schemas for ML inference service results."""

from pydantic import BaseModel, Field


class FaceEmbeddingResult(BaseModel):
    """Result of face embedding inference.

    Attributes:
        face_detected: Whether a face was found in the input image
        embedding: L2-normalized face embedding vector (zero vector when no face detected)
        embedding_dim: Dimension of the embedding vector
        det_score: Detector confidence for the embedded face (None when no face).
        face_sharpness: Variance-of-Laplacian of the detected face region, a focus
            metric (None when no face). Lets a caller flag blurry inputs without a
            second detection pass — a crisp face scores high, a blurry one low.
    """

    face_detected: bool
    embedding: list[float]
    embedding_dim: int
    det_score: float | None = None
    face_sharpness: float | None = None


class FaceMatchResult(BaseModel):
    """Result of a 1:1 comparison done entirely inside the ML service.

    Embedding both sides and scoring them in one call keeps the 512-float
    vectors off the wire: the caller only needs the similarity, so serializing
    two embeddings across an HTTP hop just to compute a dot product on the other
    side is work with no consumer.

    Attributes:
        score: Cosine similarity of the two embeddings in [-1, 1], or None when
            either side had no detectable face.
        probe_face_detected: Whether a face was found in the probe image
        reference_face_detected: Whether a face was found in the reference image
        probe_det_score: Detector confidence for the probe face (None when absent)
        reference_det_score: Detector confidence for the reference face
        probe_face_sharpness: Variance-of-Laplacian of the probe face region;
            populated only when the caller asks for it (document path).
    """

    score: float | None = None
    probe_face_detected: bool
    reference_face_detected: bool
    probe_det_score: float | None = None
    reference_det_score: float | None = None
    probe_face_sharpness: float | None = None


class FaceDetectionResult(BaseModel):
    """Result of a lightweight face-detection-only pass (no recognition).

    Attributes:
        face_detected: Whether a face was found
        bbox: [x1, y1, x2, y2] of the selected face, normalized to [0, 1]
              relative to the input image (all zeros when no face)
        score: Detector confidence for the selected face
    """

    face_detected: bool
    bbox: list[float]
    score: float = 0.0


class SpoofDetectionResult(BaseModel):
    """Result of spoof detection inference.

    Attributes:
        is_live: True if image is genuine/live, False if spoofed
        probability: Probability of image being live in [0, 1] range
        reason: None when the model scored the face, else why it could not:
            "face_too_small" — the face is below the size MiniFASNet can judge,
            so ``is_live`` is False without a model verdict (ask for a retake).
    """

    is_live: bool
    probability: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class NSFWDetectionResult(BaseModel):
    """Result of NSFW detection inference.

    Attributes:
        is_safe: True if content is safe, False if NSFW detected
        probability: Probability of content being safe in [0, 1] range
    """

    is_safe: bool
    probability: float = Field(ge=0.0, le=1.0)


class BlurDetectionResult(BaseModel):
    """Result of blur detection inference.

    Attributes:
        is_sharp: True if image is sharp, False if blurry
        probability: Probability of image being sharp in [0, 1] range
    """

    is_sharp: bool
    probability: float = Field(ge=0.0, le=1.0)


class ImageQualityAssessmentResult(BaseModel):
    """Combined result of image quality assessment across multiple checks.

    Runs three quality checks on an image: NSFW detection, spoof detection,
    and blur detection. Returns aggregated result.

    Attributes:
        passed: True if every check that ran passed, False if any of them failed
        nsfw: NSFW detection result
        spoof: Spoof detection result
        blur: Blur detection result, or None when the blur model is unavailable.
            Blur is the one optional check: the service keeps serving NSFW and
            spoof rather than failing the whole assessment, and ``passed`` is
            then decided without it.
    """

    passed: bool
    nsfw: NSFWDetectionResult
    spoof: SpoofDetectionResult
    blur: BlurDetectionResult | None = None


class DresscodeCheckResult(BaseModel):
    """Result of the ML service's uniform (dress-code) check on one image.

    Carries the raw signals only — the decision threshold is applied by the API
    layer, so a per-request threshold never needs a second inference call.

    Attributes:
        face_detected: Whether a face anchored the torso region
        region: "torso" (face-anchored) or "fallback_full_image" (no face found).
            Fallback results are materially weaker — with no anchor a blue
            background lands inside the region — so callers can discount them.
        blue_coverage: Fraction of the region inside the Loadshare-blue HSV band
        logo_match: Best normalized logo template correlation in [0, 1], or None
            when the term could not be evaluated (chest band too small, or no
            template available). None means "not measured", never "not found".
        score: Decision score in [0, 1]. With the learned classifier it is the
            uniform probability; otherwise the colour/logo fusion (blue is
            renormalized when logo_match is None).
        uniform_score: Learned classifier's uniform probability, or None when
            the classifier is not loaded.
        engine: "siglip2" (learned classifier decides) or "hsv" (colour/logo rule).
        recommended_threshold: The threshold the classifier was calibrated at,
            or None for the colour/logo rule.
        reason: None on a normal scored result, else "roi_too_small" or
            "no_blue_region" (colour/logo rule only).
    """

    face_detected: bool
    region: str
    blue_coverage: float = Field(ge=0.0, le=1.0)
    logo_match: float | None = Field(None, ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    uniform_score: float | None = Field(None, ge=0.0, le=1.0)
    engine: str = "hsv"
    recommended_threshold: float | None = None
    reason: str | None = None
