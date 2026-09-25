"""Capture quality pre-check: tell the rider now if the photo will fail.

``POST /v1/quality/check`` grades one selfie — face found, big enough, lit and
sharp; T-shirt visible and sharp — and returns rider-facing warnings. The capture
pages call it right after Capture, before Submit, so a blurry or dark photo is
retaken instead of failing face match or the dress checks.
"""

from __future__ import annotations

import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from zepiris.api.image_source import resolve_image_source
from zepiris.deps import MLAsyncDep, S3FetcherDep

router = APIRouter()
log = logging.getLogger(__name__)

#: Rider-facing text per issue, and which part of the photo it is about.
MESSAGES = {
    "face_not_found": ("face", "Face not visible — look at the camera with your face in the oval."),
    "face_too_small": ("face", "Face too far — bring the phone closer."),
    "too_dark": ("face", "Too dark to see your face — move to a brighter place."),
    "too_bright": ("face", "Too bright — move out of direct sunlight or away from the lamp."),
    "face_blurry": ("face", "Face is blurry — hold the phone steady and retake."),
    "shirt_not_visible": ("shirt", "T-shirt not visible — show your T-shirt down to the stomach."),
    "shirt_blurry": ("shirt", "T-shirt is blurry — hold the phone steady and retake."),
}


class QualityCheckRequest(BaseModel):
    image_b64: str | None = None
    image_s3: str | None = None


@router.post("/check")
async def quality_check(req: QualityCheckRequest, fetcher: S3FetcherDep, ml: MLAsyncDep) -> dict:
    """``ok`` is true when nothing needs fixing; otherwise ``warnings`` says what."""
    raw = await resolve_image_source(b64=req.image_b64, s3_url=req.image_s3, fetcher=fetcher, field="image")
    try:
        q = await ml.check_quality(raw)
    except httpx.HTTPStatusError as exc:
        # Pass on what the rider can act on (bad image / too big / overloaded),
        # never the ML service's internals.
        code = exc.response.status_code
        log.warning("quality check upstream %s: %s", code, exc.response.text[:300])
        status, message = (
            (400, "image_unreadable") if code in (400, 422)
            else (413, "image_too_large") if code == 413
            else (503, "quality_check_unavailable")
        )
        raise HTTPException(status_code=status, detail={"message": message}) from exc
    except httpx.HTTPError as exc:
        log.warning("quality check: ML service unreachable: %s", exc)
        raise HTTPException(status_code=503, detail={"message": "quality_check_unavailable"}) from exc
    return {
        "requestId": str(uuid.uuid4()),
        "ok": not q.issues,
        "warnings": [
            {"code": code, "region": MESSAGES.get(code, ("photo", code))[0], "message": MESSAGES.get(code, ("photo", code))[1]}
            for code in q.issues
        ],
        "face": {"detected": q.face_detected, "width": q.face_width, "brightness": q.face_brightness, "blur": q.face_blur},
        "shirt": {"visible": q.shirt_visible, "blur": q.shirt_blur},
    }
