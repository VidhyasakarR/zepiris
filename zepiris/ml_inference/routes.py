"""API routes for ML inference microservice."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from zepiris.framing import FrameError, decode_pair_frame
from zepiris.ml_inference.deps import (
    BlurDep,
    DresscodeDep,
    FaceEmbeddingDep,
    IQADep,
    NSFWDep,
    QualityDep,
    SpoofDep,
)
from zepiris.ml_inference.embedding_cache import reference_digest
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    CaptureQualityResult,
    DresscodeCheckResult,
    FaceDetectionResult,
    FaceEmbeddingResult,
    FaceMatchResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)

router = APIRouter()

#: Refuse images whose header declares more pixels than this. A tiny, highly
#: compressible PNG can declare 20000×20000 and decode to >1 GB; phone selfies
#: are ≤ 50 MP, and the pipeline downsizes to ≤ 1600 px anyway.
MAX_IMAGE_PIXELS = 50_000_000


def _check_pixel_budget(raw: bytes, field: str = "image") -> None:
    """Read only the image header and reject decompression bombs before decoding."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        # Pillow's own ceiling (~179 MP) trips before ours can: same answer
        raise HTTPException(status_code=413, detail=f"image_too_many_pixels: {field}") from None
    except Exception:
        # Fail closed: OpenCV decodes formats Pillow can't size up (HDR, PFM, ...),
        # and those would skip the budget. Rider photos are JPEG / PNG / WebP.
        raise HTTPException(status_code=400, detail=f"failed_to_decode_image: {field}" if field != "image" else "failed_to_decode_image") from None
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail=f"image_too_many_pixels: {field} is {w}x{h}")

# libjpeg native subsampled-decode flags — built once at import time, not per call.
_JPEG_REDUCE_FLAGS: dict[int, int] = {
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}


class ImagePayload(BaseModel):
    """Base64-encoded image payload."""

    image_b64: str


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64-encoded image to numpy array in RGB format.

    Args:
        image_b64: Base64-encoded image string

    Returns:
        np.ndarray: Image in RGB format, shape (H, W, 3), dtype uint8

    Raises:
        HTTPException: If decoding fails
    """
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_base64: {str(e)}") from e

    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty_image_data")
    _check_pixel_budget(image_bytes)

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_image_format: {str(e)}") from e

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="failed_to_decode_image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _decode_image_bytes(raw: bytes, *, field: str, reduction: int = 0) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/…) to an RGB array.

    The binary match path hands the original upload straight through, so this is
    the only decode in the whole pipeline for that image — no base64, no
    intermediate re-encode.

    ``reduction`` enables libjpeg's native subsampled decode for JPEG inputs:
    the DCT is computed at a smaller size, so CPU work falls proportionally
    rather than decoding full resolution and then resizing down.

        reduction=0  full decode (default, safe for all formats)
        reduction=2  decode at 1/2 native size  (~¼ CPU vs full)
        reduction=4  decode at 1/4 native size  (~1/16 CPU vs full)
        reduction=8  decode at 1/8 native size  (only for extreme downscaling)

    The flag is silently ignored by OpenCV for non-JPEG formats (PNG, WebP, …),
    so passing it unconditionally is always safe — the result is just the full
    image for those formats. At reduction=2 with a 256-px detector the delivered
    resolution is still well above what the recognition crop needs (112 px), so
    match quality is unaffected. Validate with ``scripts/prod_replay.py`` before
    enabling reduction=4 or higher.
    """
    if not raw:
        raise HTTPException(status_code=400, detail=f"empty_image: {field}")
    _check_pixel_budget(raw, field)
    flag = _JPEG_REDUCE_FLAGS.get(reduction, cv2.IMREAD_COLOR)
    image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), flag)
    if image_bgr is None and flag != cv2.IMREAD_COLOR:
        # Reduced decode can fail on encodings libjpeg will not scale (some
        # progressive JPEGs, unusual chroma sub-sampling). Retry at full size
        # before rejecting the image: a format we cannot subsample is still a
        # format we can decode.
        image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"failed_to_decode_image: {field}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Use /readyz to decide about sending traffic."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request):
    """Readiness: models loaded *and* warmed, so this instance can serve at speed.

    Distinct from ``/healthz`` because a freshly started instance accepts
    connections long before it can answer quickly — the first inference through a
    cold ONNX session pays one-off setup that a real request should not. An
    autoscaling group that routes on liveness alone sends the burst it just scaled
    out for straight into instances that are not ready for it.

    Point the load balancer's health check here.
    """
    state = request.app.state
    if getattr(state, "face_embedding_service", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "face_embedding_model_unavailable"},
        )
    # Liveness is a security gate: an instance told to serve it but without the
    # model would 503 every facematch, so it must not take traffic.
    if getattr(state, "liveness_required", False) and getattr(state, "spoof_service", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "spoof_model_unavailable"},
        )
    if not getattr(state, "warmed_up", False):
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "warming_up"}
        )
    return {"status": "ready"}


@router.get("/metrics")
def metrics(request: Request) -> dict:
    """Saturation, for autoscaling and dashboards.

    ``queue_depth`` is the signal to scale on. CPU utilization saturates near
    100% whether the service is comfortably busy or badly overloaded, and
    ``active`` is capped by the limiter for the same reason — neither separates
    the two. A non-zero queue means requests are waiting for CPU that does not
    exist on this instance yet, which is exactly when another one is needed.
    """
    limiter = request.app.state.inference_limiter
    service = getattr(request.app.state, "face_embedding_service", None)
    cache = getattr(service, "reference_cache", None)
    return {
        "inference": limiter.snapshot(),
        "reference_cache": cache.snapshot() if cache is not None else {"enabled": False},
        "ready": bool(getattr(request.app.state, "warmed_up", False)),
    }


@router.post("/v1/face/match", response_model=FaceMatchResult)
async def match_faces(
    request: Request,
    service: FaceEmbeddingDep,
    want_probe_sharpness: bool = False,
) -> FaceMatchResult:
    """Score one 1:1 pair from raw image bytes — the whole hot path in one call.

    The body is both original images end to end behind a 4-byte length prefix,
    not multipart. Multipart spools any part over 1 MB to a temporary **file**,
    so every request carrying a normal phone photo wrote to disk and read it back
    — pointless I/O on a service that persists nothing, and at high request rates
    a genuine source of disk churn. A framed body stays in memory and skips
    boundary scanning entirely.

    This is on top of what the single-call design already removed per side: a
    decode, a lossless PNG re-encode (~30 ms and ~10x the bytes), base64 both
    ways, and a second HTTP round trip. Only the similarity comes back, so the
    512-float vectors never touch JSON.

    Concurrency is bounded by the service's inference limiter; callers past the
    limit wait briefly and are then shed with 503 rather than queueing past their
    own timeout.
    """
    try:
        probe_raw, reference_raw = decode_pair_frame(await request.body())
    except FrameError as exc:
        raise HTTPException(status_code=400, detail=f"malformed_frame: {exc}") from exc

    limiter = request.app.state.inference_limiter
    async with limiter.slot():
        # Decided out here, where the limiter's occupancy is visible: embedding
        # the two sides at once is only free while cores are idle.
        parallel = _may_embed_in_parallel(request.app.state, limiter)
        jpeg_reduction = getattr(request.app.state, "jpeg_decode_reduction", 0)
        return await run_in_threadpool(
            _match_sync,
            service,
            probe_raw,
            reference_raw,
            want_probe_sharpness,
            parallel,
            jpeg_reduction,
        )


def _may_embed_in_parallel(state, limiter) -> bool:
    """Whether this request may embed its two sides at once.

    The test is not "is there a free slot" but "would doubling every in-flight
    request still fit". Each parallel request occupies two cores instead of one,
    so the safe condition is ``active * 2 <= limit``. A looser rule that only
    asked for a couple of spare slots measurably backfired: at 5 concurrent
    requests on 8 cores it let all five fan out to ten threads, and p50 went
    *up* (564 ms -> 658 ms) because every request then fought for a core.

    ``waiting == 0`` is the second guard — anything queued means the CPU is
    already oversubscribed, whatever the active count says.
    """
    if not getattr(state, "parallel_pair_embed", False):
        return False
    if limiter.waiting:
        return False
    return limiter.active * 2 <= limiter.limit


def _match_sync(
    service,
    probe_raw: bytes,
    reference_raw: bytes,
    want_probe_sharpness: bool,
    parallel: bool = False,
    jpeg_reduction: int = 0,
) -> FaceMatchResult:
    """Decode + embed + score, off the event loop.

    The reference is keyed by its own bytes before anything is decoded, and it is
    handed to ``match_pair`` as a **callable** rather than an array. On a cache
    hit those bytes are never decoded at all, saving a full JPEG decode and the
    BGR->RGB conversion on the majority of requests — at the hit rate measured in
    production, roughly two thirds of them. Deciding here instead (peek the
    cache, skip the decode) would be a race: the entry can be evicted between the
    peek and ``match_pair``'s own lookup, leaving nothing to embed. The callable
    also puts the decode on the thread that does the embedding, so the parallel
    path overlaps it with the probe.

    ``jpeg_reduction`` is forwarded to :func:`_decode_image_bytes` to enable
    libjpeg's native subsampled decode — see that function's docstring.
    """
    cache = getattr(service, "reference_cache", None)
    reference_key = reference_digest(reference_raw) if cache is not None and cache.enabled else None
    probe_rgb = _decode_image_bytes(probe_raw, field="probe", reduction=jpeg_reduction)
    return service.match_pair(
        probe_rgb,
        lambda: _decode_image_bytes(
            reference_raw, field="reference", reduction=jpeg_reduction
        ),
        want_probe_sharpness=want_probe_sharpness,
        reference_key=reference_key,
        parallel=parallel,
    )


@router.post("/v1/iqa/nsfw_check", response_model=NSFWDetectionResult)
def detect_nsfw(
    service: NSFWDep,
    payload: ImagePayload,
) -> NSFWDetectionResult:
    """Run NSFW detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/iqa/spoof_check", response_model=SpoofDetectionResult)
def detect_spoof(
    service: SpoofDep,
    payload: ImagePayload,
) -> SpoofDetectionResult:
    """Run spoof detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/liveness/check", response_model=SpoofDetectionResult)
async def liveness_check(request: Request, service: SpoofDep) -> SpoofDetectionResult:
    """Liveness (anti-spoof) on raw image bytes — the facematch gate.

    Takes the original upload as the body (``application/octet-stream``), like
    ``/v1/face/match``, so the probe is never base64-encoded on the hot path.
    Held to the same inference limiter as matching: it runs a face detection
    plus the MiniFASNet ensemble, and must shed under overload rather than queue.
    """
    raw = await request.body()
    limiter = request.app.state.inference_limiter
    async with limiter.slot():
        return await run_in_threadpool(
            lambda: service.forward(_decode_image_bytes(raw, field="probe"))
        )


@router.post("/v1/iqa/blur_check", response_model=BlurDetectionResult)
def detect_blur(
    service: BlurDep,
    payload: ImagePayload,
) -> BlurDetectionResult:
    """Run blur detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/face/embed", response_model=FaceEmbeddingResult)
def embed_face(
    service: FaceEmbeddingDep,
    payload: ImagePayload,
) -> FaceEmbeddingResult:
    """Generate face embedding from an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.embed(image_rgb)


@router.post("/v1/face/detect", response_model=FaceDetectionResult)
def detect_face(
    service: FaceEmbeddingDep,
    payload: ImagePayload,
) -> FaceDetectionResult:
    """Detect the primary face and return its normalized bounding box (no recognition)."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.detect_box(image_rgb)


@router.post("/v1/iqa/assess", response_model=ImageQualityAssessmentResult)
def assess_image_quality(
    service: IQADep,
    payload: ImagePayload,
) -> ImageQualityAssessmentResult:
    """Run combined image quality assessment (NSFW + spoof + blur in parallel)."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.assess(image_rgb)


@router.post("/v1/dresscode/check", response_model=DresscodeCheckResult)
async def dresscode_check(request: Request, payload: ImagePayload, service: DresscodeDep):
    """Score whether the image shows the blue Loadshare uniform shirt.

    Returns raw signals only — the decision threshold is applied by the API
    layer, so a per-request threshold costs no second inference call.

    The work is a face detection, the learned classifier (two SigLIP2 passes,
    batched) and a colour/logo pass — CPU/GPU-bound, so it runs in a worker
    thread and holds an inference-limiter slot like matching does.
    """
    # decode in the worker thread too: a big JPEG must not stall the event loop
    async with request.app.state.inference_limiter.slot():
        return await run_in_threadpool(lambda: service.check(_decode_base64_image(payload.image_b64)))


@router.post("/v1/quality/check", response_model=CaptureQualityResult)
async def quality_check(request: Request, payload: ImagePayload, service: QualityDep):
    """Capture quality: face found / big enough / lit / sharp, T-shirt visible / sharp.

    One face detection plus up to two blur-model passes (face, T-shirt), so it
    holds an inference-limiter slot like the other model routes.
    """
    # decode in the worker thread too: a big JPEG must not stall the event loop
    async with request.app.state.inference_limiter.slot():
        return await run_in_threadpool(lambda: service.check(_decode_base64_image(payload.image_b64)))
