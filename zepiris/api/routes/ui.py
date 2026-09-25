"""Operator/rider pages, served by the API so they call it same-origin (no CORS).

* ``/ui``         — operator test page: pick checks, source and selfie, then run.
* ``/ui/selfie``  — camera-only page for riders. Opened with a **POST** whose body
  carries the checkpoint parameters (``checks``, ``source_selfie_s3`` or
  ``source_selfie_b64``, optional thresholds, camera ``zoom`` and liveness
  ``challenge``), as a form
  or JSON. Nothing goes in
  the URL and nothing is configurable on screen; the page sends these values in
  the ``/v1/checkpoint/verify`` request body. A plain GET shows how to open it.
* ``/ui/static/`` — the capture guide both pages share.

Excluded from the OpenAPI schema: tools for people, not part of the API contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from zepiris.deps import SettingsDep
from zepiris.schemas.checkpoint import ALL_CHECKS

_DIR = Path(__file__).resolve().parent.parent / "ui"
#: The only files /ui/static serves — an allow-list, never a directory listing.
_STATIC = {"guide.js": "text/javascript", "guide.css": "text/css"}
#: Placeholder in selfie.html replaced by the page's parameters.
_CONFIG_SLOT = '<script id="config" type="application/json">null</script>'
_THRESHOLDS = ("threshold", "dress_color_threshold", "logo_threshold")
#: Active liveness before capture (runs in the page; the server's passive
#: liveness still decides at the checkpoint).
_CHALLENGES = ("none", "blink", "turn", "random")
#: A base64 source selfie at the API's 5 MB image cap, plus slack.
_MAX_SOURCE_B64 = 7_500_000

#: The pages and their script change together; make browsers revalidate every
#: load so a phone never runs a stale guide.js against a newer page.
_NO_CACHE = {"Cache-Control": "no-cache"}

router = APIRouter()


@router.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_DIR / "index.html", media_type="text/html", headers=_NO_CACHE)


def _render_selfie(config: dict | None) -> HTMLResponse:
    page = (_DIR / "selfie.html").read_text()
    # JSON inside <script>: escape <, > and & so no value can close the tag.
    blob = json.dumps(config).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = page.replace(_CONFIG_SLOT, f'<script id="config" type="application/json">{blob}</script>')
    # The body may carry a rider's enrolled selfie: never cache it.
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@router.get("/ui/selfie", include_in_schema=False)
async def ui_selfie_get() -> HTMLResponse:
    return _render_selfie(None)


def _selfie_config(raw: dict, allow_threshold_override: bool = False) -> dict:
    """Validate the POSTed parameters into the page's checkpoint request fields."""
    checks = raw.get("checks") or ",".join(ALL_CHECKS)
    if isinstance(checks, str):
        checks = [c.strip() for c in checks.split(",") if c.strip()]
    if not isinstance(checks, list) or not checks:
        raise HTTPException(status_code=422, detail="checks must be a non-empty list or comma-separated string")
    unknown = [c for c in checks if c not in ALL_CHECKS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown checks: {unknown}")
    cfg: dict = {"checks": list(dict.fromkeys(checks))}

    s3 = str(raw.get("source_selfie_s3") or "").strip()
    b64 = str(raw.get("source_selfie_b64") or "").strip()
    if s3 and b64:
        raise HTTPException(status_code=422, detail="give source_selfie_s3 or source_selfie_b64, not both")
    if s3:
        if not s3.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="source_selfie_s3 must be an http(s) URL")
        cfg["source_selfie_s3"] = s3
    if b64:
        if len(b64) > _MAX_SOURCE_B64:
            raise HTTPException(status_code=413, detail="source_selfie_b64 is too large")
        cfg["source_selfie_b64"] = b64
    if "face_match" in cfg["checks"] and not (s3 or b64):
        raise HTTPException(status_code=422, detail="face_match needs source_selfie_s3 or source_selfie_b64")

    def number(k: str, lo: float, hi: float) -> float | None:
        v = raw.get(k)
        if v in (None, ""):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{k} must be a number") from None
        if not math.isfinite(f) or not lo <= f <= hi:
            raise HTTPException(status_code=422, detail=f"{k} must be between {lo} and {hi}")
        return f

    thresholds = {k: number(k, 0.05, 0.99) for k in _THRESHOLDS}
    if any(v is not None for v in thresholds.values()):
        if not allow_threshold_override:
            raise HTTPException(status_code=403, detail="threshold overrides are disabled on this server")
        cfg.update({k: v for k, v in thresholds.items() if v is not None})
    zoom = number("zoom", 0.25, 1)
    if zoom is not None:
        cfg["zoom"] = zoom
    challenge = str(raw.get("challenge") or "").strip().lower()
    if challenge:
        if challenge not in _CHALLENGES:
            raise HTTPException(status_code=422, detail=f"challenge must be one of {list(_CHALLENGES)}")
        cfg["challenge"] = challenge
    # No "api" parameter: the page only ever talks to the server that served it,
    # so a third-party form POST cannot point the capture (and the enrolled
    # selfie) at another host.
    return cfg


@router.post("/ui/selfie", include_in_schema=False)
async def ui_selfie_post(request: Request, settings: SettingsDep) -> HTMLResponse:
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            raw = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="body is not valid JSON") from None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
    else:
        # a base64 source selfie easily exceeds Starlette's 1 MB multipart part default
        raw = dict(await request.form(max_part_size=_MAX_SOURCE_B64 + 1024))
    return _render_selfie(_selfie_config(raw, getattr(settings, "allow_threshold_override", False)))


@router.get("/ui/static/{name}", include_in_schema=False)
async def ui_static(name: str) -> FileResponse:
    media = _STATIC.get(name)
    if media is None:
        raise HTTPException(status_code=404)
    return FileResponse(_DIR / name, media_type=media, headers=_NO_CACHE)
