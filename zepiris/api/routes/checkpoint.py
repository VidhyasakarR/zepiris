"""Rider checkpoint: face match and dress code on one selfie, in one call.

The selfie is resolved once and both checks run concurrently, so the call costs
the slower of the two rather than their sum. Each block is exactly what its
standalone endpoint would return (``/v1/faces/facematch/verify`` and
``/v1/dresscode/match``) — the decisions are not re-implemented here.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Request

from zepiris.api.routes.dresscode import score_dresscode
from zepiris.api.routes.face import facematch_from_bytes, resolve_facematch_sources
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.schemas.checkpoint import CheckpointRequest

router = APIRouter()


@router.post("/verify")
async def checkpoint_verify(
    request: Request,
    req: CheckpointRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Is this the enrolled rider, and are they in uniform?

    ``isCleared`` is true only when the face matches (and passes liveness, when
    enabled) *and* the dress code matches. ``scores`` gathers the headline
    numbers — face match, liveness, shirt colour, logo — in one flat block.
    """
    request_id = str(uuid.uuid4())
    probe_raw, reference_raw = await resolve_facematch_sources(req, fetcher)
    dress_threshold = (
        req.dresscode_threshold
        if req.dresscode_threshold is not None
        else settings.dresscode_threshold
    )
    face, dress = await asyncio.gather(
        facematch_from_bytes(
            request,
            request_id=request_id,
            probe_raw=probe_raw,
            reference_raw=reference_raw,
            explicit_threshold=req.threshold,
            settings=settings,
            matcher=matcher,
            learner=learner,
        ),
        score_dresscode(
            request.app.state.ml_async, probe_raw, request_id=request_id, threshold=dress_threshold
        ),
    )
    face.pop("requestId", None)
    dress_body = dress.model_dump(by_alias=True)
    dress_body.pop("requestId", None)

    face_ok = bool(face["verificationResult"]["isMatch"])
    face_scores = face.get("scores") or {}
    return {
        "requestId": request_id,
        "isCleared": face_ok and dress.is_match,
        "scores": {
            "faceMatch": face["verificationResult"]["score"],
            "faceThreshold": face["verificationResult"]["threshold"],
            "liveness": face_scores.get("livenessScore"),
            "uniform": dress.scores.uniform,
            "dressColour": dress.scores.blue_coverage,
            "logo": dress.scores.logo_match,
            "dresscode": dress.score,
            "dresscodeThreshold": dress.threshold,
        },
        "faceMatch": face,
        "dresscode": dress_body,
    }
