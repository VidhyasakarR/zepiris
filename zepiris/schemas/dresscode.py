"""Public request/response contract for ``POST /v1/dresscode/match``."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DresscodeMatchRequest(BaseModel):
    """JSON body for ``POST /v1/dresscode/match``.

    Supply exactly one image input: inline base64 (a bare payload or a ``data:``
    URI) or an S3/HTTP URL. ``threshold`` overrides the configured default for
    this request only.
    """

    image_b64: str | None = None
    image_s3: str | None = None
    # bounded + finite: a negative or NaN threshold would pass anything
    threshold: float | None = Field(None, ge=0.05, le=0.99, allow_inf_nan=False)


class DresscodeScores(BaseModel):
    """The two independent signals behind the decision.

    ``logo_match`` is the only term that discriminates *Loadshare* from *blue*.
    It is ``null`` when it could not be measured — a chest band too small for the
    logo to survive — which is not the same as a logo that was looked for and
    not found.
    """

    #: Learned classifier's probability that this is the Loadshare uniform
    #: (null when the service runs the colour/logo rule only).
    uniform: float | None = None
    #: Fraction of the scored region inside the Loadshare-blue HSV band.
    blue_coverage: float = Field(..., alias="blueCoverage")
    #: Best normalized logo template correlation in [0, 1], or null if unmeasured.
    logo_match: float | None = Field(None, alias="logoMatch")

    model_config = {"populate_by_name": True}


class DresscodeMatchResponse(BaseModel):
    """Response for the dress-code match endpoint.

    ``is_match`` means "wearing a Loadshare-blue shirt". The colour term cannot
    tell brands apart, so a navy polo or a blue wall filling the region will
    also score high; read ``scores.logoMatch`` for brand confidence.
    """

    request_id: str = Field(..., alias="requestId")
    is_match: bool = Field(..., alias="isMatch")
    score: float
    threshold: float
    #: "siglip2" — the learned classifier decided; "hsv" — the colour/logo rule.
    engine: str = "hsv"
    #: Per-check model probabilities ("dress_color", "logo") and their
    #: calibrated thresholds; empty when the learned model is not loaded.
    check_scores: dict[str, float] = Field(default_factory=dict, alias="checkScores")
    check_thresholds: dict[str, float] = Field(default_factory=dict, alias="checkThresholds")
    scores: DresscodeScores
    face_detected: bool = Field(..., alias="faceDetected")
    #: "torso" (face-anchored) or "fallback_full_image" (no face found).
    region: str
    #: null on a normal result, else "roi_too_small" or "no_blue_region".
    reason: str | None = None

    model_config = {"populate_by_name": True}
