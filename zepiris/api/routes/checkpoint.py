"""Rider checkpoint: the caller chooses which checks run on one selfie.

``checks`` picks any of ``face_match``, ``dress_color`` and ``logo``; only those
run, and ``isCleared`` is true only when every requested check passes.

* face_match  — the ``/v1/faces/facematch/verify`` decision (liveness-gated when
  ``ZEPIRIS_LIVENESS_ENABLED``); needs ``source_selfie_*``.
* dress_color / logo — per-check heads of the dress-code model. Both come from
  ONE dress-code inference, so asking for both costs no more than asking for one.

Face match and dress code run concurrently when both are requested, so the call
costs the slower of the two rather than their sum. With
``ZEPIRIS_CHECKPOINT_SINGLE_CALL`` (on by default) both are answered by one ML
call, so the selfie crosses the wire and is decoded once rather than twice. A check that is not requested
is not run at all (no source needed for dress-only; no dress-code inference for
face-only).

Two responses from the same inference:

* ``POST /verify`` — scores plus the server's pass/fail and ``isCleared``.
* ``POST /score``  — scores only, no decision: for callers that decide later
  (the rider app hands them to its go-online API). ``image.sha256`` is the
  SHA-256 of the exact photo scored, so the deciding backend can check the
  scores belong to the selfie it stores.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

from zepiris.api.image_source import resolve_image_source
from zepiris.api.routes.dresscode import dresscode_upstream_error, score_dresscode
from zepiris.api.routes.face import facematch_from_bytes, resolve_facematch_sources
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exceptions import MLInferenceTimeoutError, MLInferenceTransportError
from zepiris.schemas.checkpoint import CheckpointRequest
from zepiris.schemas.ml_inference import (
    CHECKPOINT_FAILED_CHECK_HEADER,
    CheckpointCheckResult,
    DresscodeCheckResult,
    FaceMatchResult,
)
from zepiris.services.matching import FaceMatcher, RemoteFaceMatcher

router = APIRouter()

_DRESS_CHECKS = ("dress_color", "logo")


async def _check_checkpoint(ml, probe_raw: bytes, reference_raw: bytes) -> CheckpointCheckResult:
    """One ML call for both answers, failing as the call that failed would have.

    A face-match failure is translated exactly as ``RemoteFaceMatcher`` does;
    any other error status exactly as ``score_dresscode`` does. Timeouts and
    transport errors take the matcher's mapping (both calls would have hit them).
    """
    try:
        return await ml.check_checkpoint(probe_raw, reference_raw)
    except httpx.HTTPStatusError as e:
        if e.response.headers.get(CHECKPOINT_FAILED_CHECK_HEADER) == "face_match":
            RemoteFaceMatcher._raise_for_status(e)
        raise dresscode_upstream_error(e) from e
    except httpx.TimeoutException as e:
        raise MLInferenceTimeoutError() from e
    except httpx.HTTPError as e:
        raise MLInferenceTransportError(str(e)) from e


class _CombinedMatcher(FaceMatcher):
    """Hands the face half of the combined call to the unchanged facematch flow.

    Swapping the matcher, rather than reworking ``facematch_from_bytes``, keeps
    the liveness gate, threshold, learning log and response shaping on the one
    code path both routes share.
    """

    def __init__(self, pending: asyncio.Future) -> None:
        self._pending = pending

    async def match(
        self, probe_raw: bytes, reference_raw: bytes, *, want_probe_sharpness: bool = False
    ) -> FaceMatchResult:
        return (await self._pending).match


class _CombinedDresscode:
    """Hands the dress half of the combined call to the unchanged ``score_dresscode``."""

    def __init__(self, pending: asyncio.Future) -> None:
        self._pending = pending

    async def check_dresscode(self, image: bytes) -> DresscodeCheckResult:
        return (await self._pending).dresscode


async def _infer(request, req: CheckpointRequest, settings, matcher, fetcher, learner, request_id: str) -> dict:
    """Run the requested checks once: {requested, face, dress, probe_raw}."""
    requested = req.requested()
    want_face = "face_match" in requested
    want_dress = any(c in requested for c in _DRESS_CHECKS)

    if want_face:
        probe_raw, reference_raw = await resolve_facematch_sources(req, fetcher)
    else:
        probe_raw = await resolve_image_source(
            b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
        )
        reference_raw = None

    dress_ml = request.app.state.ml_async
    combined = None
    # The combined call replaces the remote matcher's call, so it applies only
    # when the match would have gone to the ML service in the first place.
    if (
        want_face
        and want_dress
        and getattr(settings, "checkpoint_single_call", True)
        and isinstance(matcher, RemoteFaceMatcher)
    ):
        # Started once and awaited by both halves; each half sees the same
        # result or the same already-mapped error, so whichever surfaces first
        # the response is the same.
        combined = asyncio.ensure_future(_check_checkpoint(dress_ml, probe_raw, reference_raw))
        matcher = _CombinedMatcher(combined)
        dress_ml = _CombinedDresscode(combined)

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
            dress_ml,
            probe_raw,
            request_id=request_id,
            threshold=settings.dresscode_threshold,
        )
    try:
        done = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    finally:
        if combined is not None:
            # A cancelled request must not leave the ML call running unobserved.
            combined.cancel()

    dress = done.get("dress")
    if dress is not None:
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
    face = done.get("face")
    if face is not None:
        face.pop("requestId", None)
    return {"requested": requested, "face": face, "dress": dress, "probe_raw": probe_raw}


def _reject_overrides(req: CheckpointRequest, settings) -> None:
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
    _reject_overrides(req, settings)
    r = await _infer(request, req, settings, matcher, fetcher, learner, request_id)
    requested, face, dress = r["requested"], r["face"], r["dress"]

    checks: dict[str, dict] = {}
    body: dict = {"requestId": request_id, "checksRequested": list(requested)}

    if face is not None:
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

    if dress is not None:
        per = dress.check_scores or {}
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


@router.post("/score")
async def checkpoint_score(
    request: Request,
    req: CheckpointRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Scores only, no pass/fail: the caller's backend decides later.

    Threshold fields in the body are ignored here (there is no decision to
    apply them to).
    """
    request_id = str(uuid.uuid4())
    r = await _infer(request, req, settings, matcher, fetcher, learner, request_id)
    requested, face, dress, probe_raw = r["requested"], r["face"], r["dress"], r["probe_raw"]

    scores: dict[str, dict] = {}
    if face is not None:
        scores["face_match"] = {
            # cosine similarity to the enrolled selfie; null when no face was found
            "similarity": face["verificationResult"]["score"],
            # probability the selfie is a live person (null when liveness is off)
            "liveness": (face.get("scores") or {}).get("livenessScore"),
            "faceDetected": bool(face.get("faceDetected")),
        }
    if dress is not None:
        per = dress.check_scores or {}
        for c in _DRESS_CHECKS:
            if c in requested:
                scores[c] = {"score": per[c]}
        if "logo" in requested and dress.logo_text is not None:
            scores["logo"]["read"] = dress.logo_text  # what was read on the shirt

    return {
        "requestId": request_id,
        "scoredAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checksRequested": list(requested),
        "scores": scores,
        "image": {"sha256": hashlib.sha256(probe_raw).hexdigest(), "bytes": len(probe_raw)},
    }
