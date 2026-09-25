"""Rider checkpoint: the caller chooses which checks run on one selfie.

``checks`` picks any of ``face_match``, ``dress_color`` and ``logo``; only those
run, and ``isCleared`` is true only when every requested check passes.

* face_match  — the ``/v1/faces/facematch/verify`` decision (liveness-gated when
  ``ZEPIRIS_LIVENESS_ENABLED``); needs ``source_selfie_*``.
* dress_color / logo — per-check heads of the dress-code model. Both come from
  ONE dress-code inference, so asking for both costs no more than asking for one.

Face match and dress code run concurrently when both are requested, so the call
costs the slower of the two rather than their sum. A check that is not requested
is not run at all (no source needed for dress-only; no dress-code inference for
face-only).
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request

from zepiris.api.image_source import resolve_image_source
from zepiris.api.routes.dresscode import score_dresscode
from zepiris.api.routes.face import facematch_from_bytes, resolve_facematch_sources
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.schemas.checkpoint import CheckpointRequest

router = APIRouter()

_DRESS_CHECKS = ("dress_color", "logo")


@router.post("/verify")
async def checkpoint_verify(
    request: Request,
    req: CheckpointRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Run the requested checks on the live selfie; cleared only if all pass."""
    request_id = str(uuid.uuid4())
    requested = req.requested()
    overrides = {"threshold": req.threshold, "dress_color_threshold": req.dress_color_threshold,
                 "logo_threshold": req.logo_threshold}
    sent = [k for k, v in overrides.items() if v is not None]
    if sent and not getattr(settings, "allow_threshold_override", False):
        # Pass marks are the server's decision; a device must not choose its own.
        raise HTTPException(
            status_code=403,
            detail={"message": "threshold_override_disabled", "fields": sent,
                    "hint": "Set ZEPIRIS_ALLOW_THRESHOLD_OVERRIDE=true to allow (testing only)."},
        )
    want_face = "face_match" in requested
    want_dress = any(c in requested for c in _DRESS_CHECKS)

    if want_face:
        probe_raw, reference_raw = await resolve_facematch_sources(req, fetcher)
    else:
        probe_raw = await resolve_image_source(
            b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
        )
        reference_raw = None

    tasks = {}
    if want_face:
        tasks["face"] = facematch_from_bytes(
            request,
            request_id=request_id,
            probe_raw=probe_raw,
            reference_raw=reference_raw,
            explicit_threshold=req.threshold,
            settings=settings,
            matcher=matcher,
            learner=learner,
        )
    if want_dress:
        tasks["dress"] = score_dresscode(
            request.app.state.ml_async,
            probe_raw,
            request_id=request_id,
            threshold=settings.dresscode_threshold,
        )
    done = dict(zip(tasks, await asyncio.gather(*tasks.values())))

    checks: dict[str, dict] = {}
    body: dict = {"requestId": request_id, "checksRequested": list(requested)}

    if want_face:
        face = done["face"]
        face.pop("requestId", None)
        result = face["verificationResult"]
        checks["face_match"] = {
            "passed": bool(result["isMatch"]),
            "score": result["score"],
            "threshold": result["threshold"],
            "liveness": (face.get("scores") or {}).get("livenessScore"),
            "livenessFailed": bool(face.get("livenessFailed")),
            "faceDetected": bool(face.get("faceDetected")),
        }
        body["faceMatch"] = face

    if want_dress:
        dress = done["dress"]
        per = dress.check_scores or {}
        if any(c in requested and c not in per for c in _DRESS_CHECKS):
            # The per-check heads live on the dress-code model; without it there
            # is no score to decide colour or logo on, and guessing would pass
            # or fail riders on noise.
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "dresscode_model_unavailable",
                    "hint": "dress_color / logo need the SigLIP2 dress-code model on the ML service "
                    "(models/siglip2_base_vision.onnx).",
                },
            )
        overrides = {"dress_color": req.dress_color_threshold, "logo": req.logo_threshold}
        for c in _DRESS_CHECKS:
            if c not in requested:
                continue
            threshold = overrides[c] if overrides[c] is not None else dress.check_thresholds[c]
            checks[c] = {"passed": per[c] >= threshold, "score": per[c], "threshold": threshold}
            if c == "logo" and dress.logo_text is not None:
                checks[c]["read"] = dress.logo_text  # what OCR read on the shirt, and why
        dress_body = dress.model_dump(by_alias=True)
        dress_body.pop("requestId", None)
        body["dresscode"] = dress_body

    body["isCleared"] = all(c["passed"] for c in checks.values())
    body["checks"] = checks
    return body
