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

With ``ZEPIRIS_CHECKPOINT_QUALITY_GATE`` (on by default) ``/verify`` first
runs a quality gate on the selfie — capture quality (face found / big enough /
lit / sharp, T-shirt visible / sharp) and, with ``ZEPIRIS_LIVENESS_ENABLED``,
liveness, concurrently. A selfie that fails the gate is not cleared and face
match and dress code are not run: a blurry or spoofed photo would only produce
scores nobody should trust, and stopping early spares two inferences.

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
from zepiris.api.routes.face import _check_liveness, facematch_from_bytes, resolve_facematch_sources
from zepiris.api.routes.quality import fetch_capture_quality, quality_warnings
from zepiris.services.matching import ProbeImageDecodeError
from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exceptions import MLInferenceTimeoutError, MLInferenceTransportError
from zepiris.schemas.checkpoint import CheckpointRequest
from zepiris.schemas.ml_inference import (
    CHECKPOINT_FAILED_CHECK_HEADER,
    CaptureQualityResult,
    CheckpointCheckResult,
    DresscodeCheckResult,
    FaceMatchResult,
    SpoofDetectionResult,
)
from zepiris.services.matching import FaceMatcher, RemoteFaceMatcher

router = APIRouter()

_DRESS_CHECKS = ("dress_color", "logo")

#: Capture-quality issues about the T-shirt; they only matter when a dress
#: check was requested, so a face-only checkpoint is not failed on them.
_SHIRT_ISSUES = ("shirt_not_visible", "shirt_blurry")


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


async def _resolve_sources(req: CheckpointRequest, fetcher) -> tuple[bytes, bytes | None]:
    """The probe, plus the reference when face_match is requested."""
    if "face_match" in req.requested():
        return await resolve_facematch_sources(req, fetcher)
    probe_raw = await resolve_image_source(
        b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
    )
    return probe_raw, None


async def _quality_gate(
    ml, probe_raw: bytes, requested, *, liveness_enabled: bool, quality_enabled: bool = True
) -> dict:
    """Capture quality and liveness (each when enabled) on the probe, concurrently.

    Returns ``{"checks": {...}, "passed": bool, "liveness": SpoofDetectionResult | None}``.
    Both calls fail closed with the same errors as their standalone routes: a
    gate that could not run must not let the selfie through.
    """
    want_dress = any(c in requested for c in _DRESS_CHECKS)
    calls = []
    if quality_enabled:
        calls.append(fetch_capture_quality(ml, probe_raw))
    if liveness_enabled:
        calls.append(_check_liveness(ml, probe_raw))
    # return_exceptions so one failing call never leaves the other running
    # unobserved; errors are then raised in a fixed order (quality first) so
    # identical requests always get the same answer.
    results = list(await asyncio.gather(*calls, return_exceptions=True))
    quality = results.pop(0) if quality_enabled else None
    liveness = results.pop(0) if liveness_enabled else None
    if (isinstance(quality, HTTPException) and quality.status_code == 400) or isinstance(
        liveness, ProbeImageDecodeError
    ):
        # An unreadable selfie is an outcome, not an error — as face match
        # reports it (decodeFailed) — so the rider gets a retake, not a retry.
        return {
            "checks": {
                "quality": {
                    "passed": False,
                    "issues": ["image_unreadable"],
                    "warnings": quality_warnings(["image_unreadable"]),
                }
            },
            "passed": False,
            "liveness": None,
        }
    for result in (quality, liveness):
        if isinstance(result, BaseException):
            raise result

    checks: dict[str, dict] = {}
    if quality is not None:
        issues = [i for i in quality.issues if want_dress or i not in _SHIRT_ISSUES]
        checks["quality"] = {
            "passed": not issues,
            "issues": issues,
            "warnings": quality_warnings(issues),
            "face": {
                "detected": quality.face_detected,
                "width": quality.face_width,
                "brightness": quality.face_brightness,
                "blur": quality.face_blur,
            },
            "shirt": {"visible": quality.shirt_visible, "blur": quality.shirt_blur},
        }
    if liveness is not None:
        checks["liveness"] = {
            "passed": liveness.is_live,
            "score": liveness.probability,
            "reason": liveness.reason,
        }
    passed = all(c["passed"] for c in checks.values())
    return {"checks": checks, "passed": passed, "liveness": liveness}


async def _infer(
    request,
    req: CheckpointRequest,
    settings,
    matcher,
    fetcher,
    learner,
    request_id: str,
    *,
    probe_raw: bytes | None = None,
    reference_raw: bytes | None = None,
    liveness: SpoofDetectionResult | None = None,
) -> dict:
    """Run the requested checks once: {requested, face, dress, probe_raw}.

    ``probe_raw`` / ``reference_raw`` skip source resolution when the caller
    already resolved them; ``liveness`` is a result the quality gate already has.
    """
    requested = req.requested()
    want_face = "face_match" in requested
    want_dress = any(c in requested for c in _DRESS_CHECKS)

    if probe_raw is None:
        probe_raw, reference_raw = await _resolve_sources(req, fetcher)

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
            liveness=liveness,
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
    requested = req.requested()
    probe_raw, reference_raw = await _resolve_sources(req, fetcher)

    checks: dict[str, dict] = {}
    body: dict = {"requestId": request_id, "checksRequested": list(requested)}

    liveness = None
    quality_enabled = req.quality_check
    if getattr(settings, "checkpoint_quality_gate", False) and (quality_enabled or settings.liveness_enabled):
        gate = await _quality_gate(
            request.app.state.ml_async,
            probe_raw,
            requested,
            liveness_enabled=settings.liveness_enabled,
            quality_enabled=quality_enabled,
        )
        checks.update(gate["checks"])
        liveness = gate["liveness"]
        if not gate["passed"]:
            # Stop here: face match and dress code are not run on a selfie
            # that failed quality or liveness.
            body["failedAt"] = "quality_gate"
            body["isCleared"] = False
            body["checks"] = checks
            return body

    r = await _infer(
        request, req, settings, matcher, fetcher, learner, request_id,
        probe_raw=probe_raw, reference_raw=reference_raw, liveness=liveness,
    )
    face, dress = r["face"], r["dress"]

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
