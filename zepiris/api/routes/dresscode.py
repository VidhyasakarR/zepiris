"""Public dress-code (uniform) match endpoint."""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, HTTPException

from zepiris.api.image_source import resolve_image_source
from zepiris.deps import MLAsyncDep, S3FetcherDep, SettingsDep
from zepiris.schemas.dresscode import (
    DresscodeMatchRequest,
    DresscodeMatchResponse,
    DresscodeScores,
)

router = APIRouter()


@router.post("/match", response_model=DresscodeMatchResponse, response_model_by_alias=True)
async def dresscode_match(
    payload: DresscodeMatchRequest,
    settings: SettingsDep,
    fetcher: S3FetcherDep,
    ml: MLAsyncDep,
) -> DresscodeMatchResponse:
    """Decide whether the person in the image is wearing the Loadshare uniform.

    Accepts the image as inline base64 (bare payload or ``data:`` URI) or as an
    S3/HTTP URL — exactly one of the two. The bytes are passed through to the ML
    service untouched; this process never decodes an image.

    ``isMatch`` means "wearing a Loadshare-**blue** shirt". The colour term
    cannot tell brands apart, so read ``scores.logoMatch`` for confidence that
    it is actually the branded one.
    """
    request_id = str(uuid.uuid4())

    raw = await resolve_image_source(
        b64=payload.image_b64,
        s3_url=payload.image_s3,
        fetcher=fetcher,
        field="image",
        # A dress-code probe is a fresh capture behind a fresh URL; caching it
        # would only churn the fetcher's byte budget without ever hitting.
        cacheable=False,
    )

    threshold = payload.threshold if payload.threshold is not None else settings.dresscode_threshold
    return await score_dresscode(ml, raw, request_id=request_id, threshold=threshold)


#: Threshold for the colour/logo rule when neither the request, the settings nor
#: the ML engine supplies one.
HSV_RULE_THRESHOLD = 0.55


async def score_dresscode(
    ml, raw: bytes, *, request_id: str, threshold: float | None
) -> DresscodeMatchResponse:
    """Score one image for the uniform and apply ``threshold``.

    ``threshold=None`` uses the engine's own calibrated threshold (the learned
    classifier reports one), else the colour/logo rule's 0.55. Shared with the
    checkpoint route, which scores the facematch selfie.
    """
    try:
        result = await ml.check_dresscode(raw)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail={"message": "dresscode_upstream_error", "upstream": exc.response.text[:500]},
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "ml_service_unreachable", "error": str(exc)},
        ) from exc

    if threshold is None:
        threshold = result.recommended_threshold or HSV_RULE_THRESHOLD

    return DresscodeMatchResponse(
        request_id=request_id,
        is_match=result.score >= threshold and result.reason is None,
        score=result.score,
        threshold=threshold,
        engine=result.engine,
        scores=DresscodeScores(
            uniform=result.uniform_score,
            blue_coverage=result.blue_coverage,
            logo_match=result.logo_match,
        ),
        face_detected=result.face_detected,
        region=result.region,
        reason=result.reason,
    )
