"""Public contract for ``POST /v1/checkpoint/verify`` (face match + dress code in one call)."""

from __future__ import annotations

from pydantic import Field

from zepiris.schemas.face import FaceMatchRequest


class CheckpointRequest(FaceMatchRequest):
    """Same body as ``/v1/faces/facematch/verify``, plus a dress-code threshold.

    ``face_check_*`` is the live selfie: it is face-matched (liveness-gated when
    enabled) *and* scored for the uniform. ``source_selfie_*`` is the enrolled
    reference. Each side is exactly one of base64 (bare or ``data:`` URI) or an
    S3/HTTP URL.
    """

    dresscode_threshold: float | None = Field(
        None, description="Overrides ZEPIRIS_DRESSCODE_THRESHOLD for this request."
    )
