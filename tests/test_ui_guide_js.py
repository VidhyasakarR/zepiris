"""Runs the guide.js unit tests (tests/js/guide.test.mjs) when Node is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_guide_js_unit_tests() -> None:
    script = Path(__file__).parent / "js" / "guide.test.mjs"
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
