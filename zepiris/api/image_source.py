"""Resolve an image input (inline base64 or S3 URL) to raw, size-validated bytes.

Extracted from ``zepiris.api.routes.face`` so every route that accepts an image
shares one implementation of the base64/S3 rules, the size limit and the error
mapping. Behaviour is unchanged from the original private helpers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii

from zepiris.exceptions import (
    EmptyUploadError,
    ImageSourceError,
    ImageTooLargeError,
)
from zepiris.schemas.face import MAX_IMAGE_SIZE_BYTES, MAX_IMAGE_SIZE_MB

#: Above this payload size, base64 decoding moves to a worker thread. Decoding a
#: few hundred KB takes single-digit milliseconds, which is nothing once — and a
#: hard throughput ceiling when 100 requests do it on the event loop at the same
#: time, since none of them can make progress while one decodes. Below the
#: threshold the thread hop costs more than the decode it avoids.
B64_OFFLOAD_THRESHOLD_BYTES = 64 * 1024


def validate_image_bytes(raw: bytes) -> None:
    if not raw:
        raise EmptyUploadError()
    if len(raw) > MAX_IMAGE_SIZE_BYTES:
        mb = len(raw) / (1024 * 1024)
        raise ImageTooLargeError(mb=mb, max_mb=MAX_IMAGE_SIZE_MB)


def decode_b64_sync(value: str, *, field: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:"):
        # Strip the "data:<mime>;base64," prefix, keep the payload.
        payload = payload.partition(",")[2]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageSourceError(field=field, reason="invalid_base64") from exc
    validate_image_bytes(raw)
    return raw


async def decode_b64_image(value: str, *, field: str) -> bytes:
    """Decode an inline base64 image payload to raw, size-validated bytes.

    Accepts both a bare base64 string and a ``data:`` URI
    (e.g. ``data:image/jpeg;base64,<payload>``). Large payloads decode off the
    event loop — see :data:`B64_OFFLOAD_THRESHOLD_BYTES`.
    """
    if len(value) >= B64_OFFLOAD_THRESHOLD_BYTES:
        return await asyncio.to_thread(decode_b64_sync, value, field=field)
    return decode_b64_sync(value, field=field)


async def resolve_image_source(
    *, b64: str | None, s3_url: str | None, fetcher, field: str, cacheable: bool = False
) -> bytes:
    """Resolve one image side to raw bytes.

    Each side is supplied via exactly one of two inputs — inline base64
    (``<field>_b64``) or an S3 URL (``<field>_s3``). Supplying both is
    ambiguous and rejected (400); supplying neither is missing (400).

    Whichever way the bytes arrive, they are passed through untouched — the
    consumer decodes them exactly once, wherever the models live.

    ``cacheable`` marks a side whose URL returns the same bytes on every
    request — the enrolled reference. Only that side may be served from the
    fetcher's bytes cache; a fresh capture behind a fresh URL would only churn
    the budget.
    """
    has_b64 = bool(b64 and b64.strip())
    has_s3 = bool(s3_url and s3_url.strip())
    if has_b64 and has_s3:
        raise ImageSourceError(field=field, reason="ambiguous")
    if has_b64:
        return await decode_b64_image(b64, field=field)
    if has_s3:
        raw = await fetcher.fetch(s3_url.strip(), cacheable=cacheable)
        validate_image_bytes(raw)
        return raw
    raise ImageSourceError(field=field, reason="missing")
