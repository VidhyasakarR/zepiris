"""Regression tests for the input-hardening fixes: pixel bombs, SSRF, bounded thresholds."""

from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from zepiris.exceptions import ReferenceImageFetchError
from zepiris.ml_inference.routes import _check_pixel_budget
from zepiris.schemas.dresscode import DresscodeMatchRequest
from zepiris.schemas.face import FaceMatchRequest
from zepiris.services.s3_fetcher import S3ImageFetcher, make_url_guard


def _png(w: int, h: int) -> bytes:
    b = io.BytesIO()
    Image.new("L", (w, h)).save(b, "PNG")
    return b.getvalue()


# ---- pixel budget ------------------------------------------------------------
@pytest.mark.parametrize("side", [8000, 15000])  # 15000² is past Pillow's own bomb limit
def test_pixel_bomb_rejected(side: int) -> None:
    with pytest.raises(HTTPException) as e:
        _check_pixel_budget(_png(side, side))
    assert e.value.status_code == 413


def test_normal_photo_passes_pixel_budget() -> None:
    _check_pixel_budget(_png(1080, 1440))


def test_unreadable_header_fails_closed() -> None:
    # a format Pillow can't size up must not skip the budget
    with pytest.raises(HTTPException) as e:
        _check_pixel_budget(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 30000 +X 30000\n")
    assert e.value.status_code == 400


# ---- SSRF guard ------------------------------------------------------------------
def _guarded(handler, **guard) -> S3ImageFetcher:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True,
        event_hooks={"request": [make_url_guard(**guard)]},
    )
    return S3ImageFetcher(client=client, max_bytes=1024)


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"img")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8001/healthz",          # the ML service
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://10.0.0.5/x.jpg",
    "http://[::1]/x.jpg",
    "http://[::ffff:127.0.0.1]/x.jpg",
    "http://localhost/x.jpg",
])
def test_private_targets_refused(url: str) -> None:
    with pytest.raises(ReferenceImageFetchError) as e:
        asyncio.run(_guarded(_ok).fetch(url))
    assert e.value.detail["reason"] == "url_not_allowed"


def test_public_ip_allowed() -> None:
    assert asyncio.run(_guarded(_ok).fetch("https://52.216.0.1/rider.jpg")) == b"img"


def test_redirect_into_private_network_refused() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "52.216.0.1":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
        return httpx.Response(200, content=b"secret")

    with pytest.raises(ReferenceImageFetchError):
        asyncio.run(_guarded(handler).fetch("https://52.216.0.1/rider.jpg"))


def test_allow_list() -> None:
    f = _guarded(_ok, allowed_hosts=(".amazonaws.com",), block_private=False)
    assert asyncio.run(f.fetch("https://bucket.s3.amazonaws.com/r.jpg")) == b"img"
    with pytest.raises(ReferenceImageFetchError):
        asyncio.run(f.fetch("https://evil.example/r.jpg"))
    with pytest.raises(ReferenceImageFetchError):
        asyncio.run(f.fetch("https://amazonaws.com.evil.example/r.jpg"))


def test_oversize_body_cut_off_while_streaming() -> None:
    f = _guarded(lambda _: httpx.Response(200, content=b"x" * 5000), block_private=False)
    with pytest.raises(ReferenceImageFetchError) as e:
        asyncio.run(f.fetch("https://s3/r.jpg"))
    assert e.value.detail["reason"] == "too_large"


# ---- thresholds are bounded everywhere ----------------------------------------------
@pytest.mark.parametrize("bad", [-5, 0, 1.5, float("nan"), float("inf")])
def test_thresholds_bounded(bad: float) -> None:
    with pytest.raises(ValidationError):
        DresscodeMatchRequest(image_b64="x", threshold=bad)
    with pytest.raises(ValidationError):
        FaceMatchRequest(face_check_b64="x", threshold=bad)
