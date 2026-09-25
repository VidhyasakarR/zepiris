"""UI pages: served, POST-configured rider page, shared assets on an allow-list."""

import json
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import ui
from zepiris.deps import SettingsDep


class _Settings:
    allow_threshold_override = False


def _client(*, overrides: bool = False) -> TestClient:
    app = FastAPI()
    app.include_router(ui.router)
    settings = _Settings()
    settings.allow_threshold_override = overrides
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: settings
    return TestClient(app)


def _config(html: str):
    m = re.search(r'<script id="config" type="application/json">(.*?)</script>', html, re.S)
    assert m, "config slot missing"
    return json.loads(m.group(1))


def test_operator_page_served() -> None:
    r = _client().get("/ui")
    assert r.status_code == 200 and "createCapture" in r.text


def test_selfie_get_has_no_config() -> None:
    r = _client().get("/ui/selfie")
    assert r.status_code == 200
    assert _config(r.text) is None


def test_selfie_post_form_embeds_params_in_page_not_url() -> None:
    r = _client(overrides=True).post(
        "/ui/selfie",
        data={"checks": "face_match,logo", "source_selfie_s3": "https://s3/rider.jpg", "logo_threshold": "0.5"},
    )
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    assert _config(r.text) == {
        "checks": ["face_match", "logo"], "source_selfie_s3": "https://s3/rider.jpg", "logo_threshold": 0.5,
    }


def test_selfie_post_json() -> None:
    r = _client().post("/ui/selfie", json={"checks": ["dress_color"]})
    assert _config(r.text) == {"checks": ["dress_color"]}


def test_selfie_post_defaults_to_all_checks() -> None:
    r = _client().post("/ui/selfie", json={"source_selfie_b64": "abc"})
    assert _config(r.text)["checks"] == ["face_match", "dress_color", "logo"]


def test_selfie_post_rejects_bad_params() -> None:
    c = _client()
    assert c.post("/ui/selfie", json={"checks": ["shoes"]}).status_code == 422
    assert c.post("/ui/selfie", json={"checks": ["face_match"]}).status_code == 422  # no source
    assert c.post("/ui/selfie", json={"checks": "logo", "source_selfie_s3": "file:///etc/passwd"}).status_code == 422
    assert _client(overrides=True).post("/ui/selfie", json={"checks": "logo", "threshold": "high"}).status_code == 422


def test_selfie_config_cannot_break_out_of_script() -> None:
    evil = "https://s3/x.jpg</script><script>alert(1)</script>"
    r = _client().post("/ui/selfie", json={"checks": ["face_match"], "source_selfie_s3": evil})
    assert "</script><script>alert(1)" not in r.text
    assert _config(r.text)["source_selfie_s3"] == evil


def test_shared_assets_served_with_types() -> None:
    c = _client()
    assert c.get("/ui/static/guide.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/ui/static/guide.css").headers["content-type"].startswith("text/css")


def test_static_is_an_allow_list() -> None:
    c = _client()
    for name in ("index.html", "../routes/ui.py", "selfie.html", "nope.js"):
        assert c.get(f"/ui/static/{name}").status_code == 404


def test_selfie_zoom_param() -> None:
    c = _client()
    assert _config(c.post("/ui/selfie", json={"checks": ["logo"], "zoom": "0.8"}).text)["zoom"] == 0.8
    assert c.post("/ui/selfie", json={"checks": ["logo"], "zoom": 0.3}).status_code == 422
    assert c.post("/ui/selfie", json={"checks": ["logo"], "zoom": "wide"}).status_code == 422


def test_selfie_challenge_param() -> None:
    c = _client()
    assert _config(c.post("/ui/selfie", json={"checks": ["logo"], "challenge": "Blink"}).text)["challenge"] == "blink"
    assert c.post("/ui/selfie", json={"checks": ["logo"], "challenge": "dance"}).status_code == 422


def test_selfie_threshold_override_needs_server_opt_in() -> None:
    body = {"checks": ["logo"], "logo_threshold": 0.5}
    assert _client().post("/ui/selfie", json=body).status_code == 403
    assert _config(_client(overrides=True).post("/ui/selfie", json=body).text)["logo_threshold"] == 0.5


def test_selfie_rejects_non_finite_numbers() -> None:
    c = _client(overrides=True)
    for bad in ("nan", "inf", "-inf", "-1", "1.5"):
        assert c.post("/ui/selfie", json={"checks": ["logo"], "logo_threshold": bad}).status_code == 422, bad
    for bad in ("nan", "inf"):
        assert c.post("/ui/selfie", json={"checks": ["logo"], "zoom": bad}).status_code == 422, bad


def test_selfie_has_no_api_redirect() -> None:
    # a third-party form must not be able to point the page at another host
    cfg = _config(_client().post("/ui/selfie", json={"checks": ["logo"], "api": "https://evil.example"}).text)
    assert "api" not in cfg


def test_selfie_accepts_large_multipart_source() -> None:
    big = "A" * 2_000_000  # > Starlette's 1 MB default part size
    r = _client().post("/ui/selfie", files={"checks": (None, "face_match"), "source_selfie_b64": (None, big)})
    assert r.status_code == 200, r.text[:200]
