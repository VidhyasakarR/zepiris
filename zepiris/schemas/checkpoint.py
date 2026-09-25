"""Public contract for ``POST /v1/checkpoint/verify`` — the checks the caller asks for."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: What the checkpoint can verify on the live selfie.
#:   face_match  — same person as the enrolled source selfie (liveness-gated when enabled)
#:   dress_color — the shirt is Loadshare blue
#:   logo        — the Loadshare print (chest logo / wordmark) is on the shirt
Check = Literal["face_match", "dress_color", "logo"]
ALL_CHECKS: tuple[Check, ...] = ("face_match", "dress_color", "logo")


class CheckpointRequest(BaseModel):
    """JSON body: the live selfie, the enrolled source (for face_match), and which checks to run.

    ``face_check_*`` is the live selfie — exactly one of base64 (bare or
    ``data:`` URI) or an S3/HTTP URL. ``source_selfie_*`` is the enrolled
    reference, required only when ``face_match`` is requested. ``checks``
    omitted = all three; only the listed checks run, and ``isCleared`` is true
    only when every one of them passes.
    """

    face_check_b64: str | None = None
    face_check_s3: str | None = None
    source_selfie_b64: str | None = None
    source_selfie_s3: str | None = None
    checks: list[Check] | None = Field(
        None, description='Subset of ["face_match", "dress_color", "logo"]; omitted = all.'
    )
    # Pass-mark overrides: rejected unless ZEPIRIS_ALLOW_THRESHOLD_OVERRIDE, and
    # always bounded (and finite) so even a trusted caller cannot disable a check.
    threshold: float | None = Field(None, ge=0.05, le=0.99, allow_inf_nan=False, description="Face-match threshold override.")
    dress_color_threshold: float | None = Field(None, ge=0.05, le=0.99, allow_inf_nan=False, description="Dress-colour threshold override.")
    logo_threshold: float | None = Field(None, ge=0.05, le=0.99, allow_inf_nan=False, description="Logo threshold override.")

    @field_validator("checks")
    @classmethod
    def _non_empty_unique(cls, v: list[Check] | None) -> list[Check] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("checks must name at least one check, or be omitted for all")
        return list(dict.fromkeys(v))  # de-duplicate, keep the caller's order

    def requested(self) -> tuple[Check, ...]:
        return tuple(self.checks) if self.checks else ALL_CHECKS
