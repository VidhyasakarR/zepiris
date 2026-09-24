"""Operator test page: capture a photo, run face match + dress code, show scores.

Served by the API itself so the page calls the endpoints same-origin — no CORS
configuration, and it works wherever the API is reachable. Excluded from the
OpenAPI schema: it is a tool for people, not part of the API contract.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

_INDEX = Path(__file__).resolve().parent.parent / "ui" / "index.html"

router = APIRouter()


@router.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html")
