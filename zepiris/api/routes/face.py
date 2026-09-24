from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from zepiris.api.image_source import (
    decode_b64_image,
    decode_b64_sync,
    resolve_image_source,
    validate_image_bytes,
)
from zepiris.deps import (
    EmbeddingDep,
    LearnerDep,
    MatcherDep,
    S3FetcherDep,
    SettingsDep,
)
from zepiris.exceptions import (
    DocumentTooBlurryError,
    FeedbackValidationError,
    MLInferenceTimeoutError,
    MLInferenceTransportError,
    ReferenceFaceNotFoundError,
)
from zepiris.schemas.face import (
    DocMatchRequest,
    FaceMatchRequest,
    VerificationResult,
    VerificationScores,
    VerifyResponse,
)
from zepiris.services.learning import FACE_KIND, GENERIC_DOC_TYPE
from zepiris.schemas.ml_inference import SpoofDetectionResult
from zepiris.services.matching import ProbeImageDecodeError, RemoteFaceMatcher

router = APIRouter()


def _resolve_threshold(
    explicit: float | None, learner, doc_type: str, default: float
) -> tuple[float, str]:
    """Resolve the decision threshold and report where it came from.

    Precedence: explicit per-request value > learned (calibrated from feedback
    for this doc_type) > configured default. The source string surfaces in the
    response so callers can see the adaptive learning being applied.
    """
    if explicit is not None:
        return explicit, "explicit"
    learned = learner.learned_threshold(doc_type)
    if learned is not None:
        return learned, "learned"
    return default, "default"


def _scores_dict(
    match_score: float | None,
    threshold: float,
    ml_struct,
    liveness: SpoofDetectionResult | None = None,
) -> dict:
    """Flat numeric summary gathered from the match score + the IQA result.

    Quality scores are ``None`` when the liveness/IQA gate did not run
    (``ml_struct is None``, e.g. docmatch). ``blurScore`` is also ``None`` on its
    own when the gate ran without the blur model — the rest of the block is
    still filled in. ``liveness`` is the standalone liveness gate's result; it
    fills ``livenessScore`` when the full IQA did not run.
    """
    blur = ml_struct.blur if ml_struct else None
    if ml_struct:
        liveness_score = ml_struct.spoof.probability
    else:
        liveness_score = liveness.probability if liveness is not None else None
    return {
        "matchScore": match_score,
        "threshold": threshold,
        "margin": (match_score - threshold) if match_score is not None else None,
        "livenessScore": liveness_score,
        "blurScore": blur.probability if blur is not None else None,
        "nsfwSafeScore": ml_struct.nsfw.probability if ml_struct else None,
    }


async def _record_outcome(learner, *, request_id: str, doc_type: str, body: dict) -> None:
    """Log a scored verification so operator feedback can calibrate thresholds.

    Only scores and the decision are logged — never images or personal data.
    Unscored results (decode/no-face failures) carry no signal for threshold
    fitting and are skipped.

    The write goes to a worker thread: it appends to a file under a process-wide
    lock, which on the event loop would serialize every request in the process
    behind one another's disk I/O — a global bottleneck on a path that is
    otherwise fully concurrent.
    """
    if not learner.enabled:
        # Disabled is the production default; skip the thread hop entirely
        # rather than paying one per request to run a no-op.
        return
    result = body.get("verificationResult") or {}
    score = result.get("score")
    if score is None:
        return
    scores = body.get("scores") or {}
    await asyncio.to_thread(
        learner.record_sample,
        request_id=request_id,
        doc_type=doc_type,
        score=float(score),
        threshold=float(result["threshold"]),
        is_match=bool(result["isMatch"]),
        liveness_score=scores.get("livenessScore"),
        blur_score=scores.get("blurScore"),
        nsfw_safe_score=scores.get("nsfwSafeScore"),
    )


# Image-source handling (base64 / S3, size limits, error mapping) is shared
# with the dresscode route; the implementation lives in
# zepiris.api.image_source. Aliased here so call sites below read unchanged.
_validate_image_bytes = validate_image_bytes
_decode_b64_sync = decode_b64_sync
_decode_b64_image = decode_b64_image
_resolve_image_source = resolve_image_source


async def _check_liveness(ml, raw: bytes) -> SpoofDetectionResult:
    """Call the ML liveness gate, mapping failures exactly as the matcher does.

    The gate fails closed: an unreachable or unloaded liveness model is a 503,
    never a silent pass — a verification that skipped the check it was
    configured to run must not come back as a match.
    """
    try:
        return await ml.check_liveness(raw)
    except httpx.HTTPStatusError as e:
        RemoteFaceMatcher._raise_for_status(e)
    except httpx.TimeoutException as e:
        raise MLInferenceTimeoutError() from e
    except httpx.HTTPError as e:
        raise MLInferenceTransportError(str(e)) from e


async def _run_match(
    *,
    request_id: str,
    probe_raw: bytes,
    reference_raw: bytes,
    matcher,
    decision_threshold: float,
    threshold_source: str,
    probe_is_document: bool = False,
    min_sharpness: float = 0.0,
    liveness_check: Callable[[], Awaitable[SpoofDetectionResult]] | None = None,
) -> dict:
    """Core 1:1 verification, scored in a single call to the ML service.

    The API never decodes or re-encodes an image on this path: the bytes that
    arrived (from base64 or S3) are the bytes the ML service receives, and only
    a similarity score comes back. Everything the old path spent per side — a
    decode, a lossless PNG re-encode, base64 both ways, and its own HTTP round
    trip — is gone, along with the 512-float vectors that used to cross the wire
    just to be dot-producted here.

    ``probe_raw`` is the image submitted *now* to be verified — a live face
    (facematch) or an ID document (docmatch). ``reference_raw`` is the user's
    enrolled selfie; it is only ever embedded.

    With ``probe_is_document`` the extracted face's sharpness is reported in
    ``documentFace``; when ``min_sharpness`` > 0 a too-blurry face is rejected.

    With ``liveness_check`` the probe's liveness runs concurrently with the
    match, so the gate adds no serial round trip; a non-live probe is reported
    with its score but ``isMatch`` false and ``livenessFailed`` true.
    """

    def _verification_result(is_match: bool, score: float | None) -> dict:
        return {
            "isMatch": is_match,
            "score": score,
            "threshold": decision_threshold,
            "thresholdSource": threshold_source,
        }

    liveness: SpoofDetectionResult | None = None
    try:
        match_coro = matcher.match(
            probe_raw, reference_raw, want_probe_sharpness=probe_is_document
        )
        if liveness_check is None:
            result = await match_coro
        else:
            # return_exceptions so a failing match never leaves the liveness call
            # running unobserved; the match error takes precedence (an
            # undecodable probe is reported as decodeFailed, not as a 503).
            result, liveness = await asyncio.gather(
                match_coro, liveness_check(), return_exceptions=True
            )
            if isinstance(result, BaseException):
                raise result
            if isinstance(liveness, BaseException):
                raise liveness
    except ProbeImageDecodeError:
        # An unreadable capture is an ordinary outcome, not a fault: report it in
        # the response body the same way the caller sees every other verdict.
        return {
            "requestId": request_id,
            "decodeFailed": True,
            "faceDetected": False,
            "iqaPassed": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, None),
        }

    doc_diag = None
    if probe_is_document:
        doc_diag = {
            "faceDetected": bool(result.probe_face_detected),
            "detScore": (
                round(result.probe_det_score, 4)
                if result.probe_det_score is not None
                else None
            ),
            "sharpness": result.probe_face_sharpness,
        }
        sharp = result.probe_face_sharpness
        low = (
            min_sharpness > 0
            and result.probe_face_detected
            and sharp is not None
            and sharp < min_sharpness
        )
        doc_diag["lowQuality"] = bool(low)
        if low:
            raise DocumentTooBlurryError(sharpness=sharp, min_sharpness=min_sharpness)

    if not result.probe_face_detected:
        return {
            "requestId": request_id,
            "imageQualityAssessment": None,
            "iqaPassed": True,
            "faceDetected": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, None),
            "documentFace": doc_diag,
        }

    if not result.reference_face_detected:
        raise ReferenceFaceNotFoundError()

    score = float(result.score)
    if liveness is not None and not liveness.is_live:
        return {
            "requestId": request_id,
            "imageQualityAssessment": None,
            "liveness": liveness.model_dump(),
            "livenessFailed": True,
            "iqaPassed": False,
            "faceDetected": True,
            "verificationResult": _verification_result(False, score),
            "scores": _scores_dict(score, decision_threshold, None, liveness),
            "documentFace": doc_diag,
        }
    return VerifyResponse(
        request_id=request_id,
        image_quality_assessment=None,
        verification_result=VerificationResult(
            is_match=bool(score >= decision_threshold),
            score=score,
            threshold=decision_threshold,
            threshold_source=threshold_source,
        ),
        scores=VerificationScores(**_scores_dict(score, decision_threshold, None, liveness)),
        liveness=liveness,
        document_face=doc_diag,
        face_detected=True,
        iqa_passed=True,
    ).model_dump(by_alias=True)


async def _resolve_two_sources(
    *, probe_kwargs: dict, reference_kwargs: dict
) -> tuple[bytes, bytes]:
    """Resolve the probe and reference images concurrently.

    Both sides are independent network work, so fetching them in parallel removes
    one S3 round-trip from the critical path when both arrive as S3 URLs. Base64
    inputs resolve instantly either way. Both fetches are awaited directly rather
    than handed to threads: at high concurrency, parking a worker thread per
    in-flight fetch would make the thread pool the bottleneck instead of S3.
    """
    probe_raw, reference_raw = await asyncio.gather(
        _resolve_image_source(**probe_kwargs),
        _resolve_image_source(**reference_kwargs),
    )
    return probe_raw, reference_raw


@router.post("/facematch/verify")
async def facematch_verify(
    request: Request,
    req: FaceMatchRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Face-to-face 1:1 verification: an incoming live face vs the enrolled selfie.

    JSON body. Incoming face being verified (``face_check``; the live capture,
    gated on liveness/quality): supply exactly one of ``face_check_b64`` (base64
    binary) or ``face_check_s3`` (S3 URL). Enrolled reference selfie / source of
    truth (``source_selfie``; only embedded, from the DB): supply exactly one of
    ``source_selfie_b64`` or ``source_selfie_s3``. Sending both inputs for a
    side is rejected.
    """
    request_id = str(uuid.uuid4())
    probe_raw, reference_raw = await resolve_facematch_sources(req, fetcher)
    return await facematch_from_bytes(
        request,
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        explicit_threshold=req.threshold,
        settings=settings,
        matcher=matcher,
        learner=learner,
    )


async def resolve_facematch_sources(req: FaceMatchRequest, fetcher) -> tuple[bytes, bytes]:
    """Resolve the (probe, reference) bytes of a facematch-shaped request body.

    Shared with the checkpoint route, which takes the same selfie/source body.
    """
    return await _resolve_two_sources(
        probe_kwargs=dict(
            b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
        ),
        reference_kwargs=dict(
            b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
            field="source_selfie", cacheable=True,
        ),
    )


async def facematch_from_bytes(
    request: Request,
    *,
    request_id: str,
    probe_raw: bytes,
    reference_raw: bytes,
    explicit_threshold: float | None,
    settings,
    matcher,
    learner,
) -> dict:
    """Facematch on already-resolved bytes: threshold, liveness gate, match, learning log.

    The body of ``/facematch/verify`` after source resolution, factored out so
    the checkpoint route runs the identical face decision alongside dress code.
    """
    decision_threshold, threshold_source = _resolve_threshold(
        explicit_threshold, learner, FACE_KIND, settings.verify_threshold
    )
    liveness_check = None
    if settings.liveness_enabled:
        # Read from app state only when the gate is on, so a deployment with
        # liveness off never touches the liveness client at all.
        ml = request.app.state.ml_async
        liveness_check = lambda: _check_liveness(ml, probe_raw)  # noqa: E731
    body = await _run_match(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        matcher=matcher,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        liveness_check=liveness_check,
    )
    await _record_outcome(learner, request_id=request_id, doc_type=FACE_KIND, body=body)
    return body


@router.post("/docmatch/verify")
async def docmatch_verify(
    req: DocMatchRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Doc-to-face 1:1 verification: the face on an uploaded ID document vs the
    enrolled selfie.

    JSON body. Incoming document being verified (``doc_check``; face
    auto-extracted, then embedded — no liveness gate, since a printed ID is not
    a live capture): supply exactly one of ``doc_check_b64`` (base64 binary) or
    ``doc_check_s3`` (S3 URL). Enrolled reference selfie / source of truth
    (``source_selfie``; only embedded, from the DB): supply exactly one of
    ``source_selfie_b64`` or ``source_selfie_s3``. Defaults to a more lenient
    threshold than face-match because printed ID photos embed weaker.

    ``doc_type`` (e.g. ``aadhaar``, ``pan``) buckets the request for adaptive
    threshold learning: each document type's threshold is calibrated separately
    from operator feedback, since Aadhaar and PAN photos score differently.
    """
    request_id = str(uuid.uuid4())
    probe_raw, reference_raw = await _resolve_two_sources(
        probe_kwargs=dict(
            b64=req.doc_check_b64, s3_url=req.doc_check_s3, fetcher=fetcher, field="doc_check"
        ),
        reference_kwargs=dict(
            b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
            field="source_selfie", cacheable=True,
        ),
    )
    doc_kind = (req.doc_type or GENERIC_DOC_TYPE).strip().lower() or GENERIC_DOC_TYPE
    decision_threshold, threshold_source = _resolve_threshold(
        req.threshold, learner, doc_kind, settings.doc_verify_threshold
    )
    body = await _run_match(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        matcher=matcher,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        probe_is_document=True,
        min_sharpness=settings.doc_min_sharpness,
    )
    await _record_outcome(learner, request_id=request_id, doc_type=doc_kind, body=body)
    return body


@router.post("/feedback")
async def verification_feedback(
    learner: LearnerDep,
    request_id: str | None = Form(None),
    genuine: bool | None = Form(None),
) -> dict:
    """Report the confirmed outcome of a past verification.

    ``genuine=true`` means the selfie and the document/reference really were the
    same person (per downstream KYC/manual review); ``genuine=false`` means an
    impostor. Feedback is joined with the logged match score and used to re-fit
    the decision threshold for that request's document type — this is how the
    system keeps improving on real Aadhaar/PAN traffic while it runs.
    """
    if not request_id or genuine is None:
        raise FeedbackValidationError("provide request_id and genuine")
    summary = learner.record_feedback(request_id=request_id, genuine=genuine)
    return {"requestId": request_id, **summary}


# Backward-compatible alias for the original single endpoint (= face match).
@router.post("/verify")
async def verify_face(
    request: Request,
    req: FaceMatchRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Deprecated alias of ``/facematch/verify`` (kept for existing callers)."""
    return await facematch_verify(
        request=request,
        req=req,
        settings=settings,
        matcher=matcher,
        fetcher=fetcher,
        learner=learner,
    )


@router.post("/detect")
async def detect_face(
    embedding_svc: EmbeddingDep,
    file: UploadFile = File(...),
) -> dict:
    """Cheap face-present check: is a face visible, and where?

    Deliberately never raises on a bad frame — callers poll this, and a poll that
    500s on an unreadable frame is harder to use than one that says "no face".
    """
    raw = await file.read()
    if not raw:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        result = embedding_svc.detect_box(image_rgb)
    except Exception:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    return {"faceDetected": bool(result.face_detected), "bbox": result.bbox}
