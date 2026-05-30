"""Shared fixtures + baseline constants for the v2 Budapest e2e suite.

See `README.md` in this directory for the per-scenario contract:
what each test does, what it asserts, what new tabs must extend.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Baseline — the contract every v2 tab/feature must keep green
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(r"D:\Budapest2025_Google")
PROFILE_NAME = "profile_4.json"

# Reference run on the user's machine. Producer = legacy `fc_app`; produced
# the cluster shape v2 must match. See README.md for full breakdown.
REFERENCE_RUN_ID = "6437d335de914755bc3edb825c9591c0"
REFERENCE_RUN_DIR = Path.home() / ".sim_bench" / "runs" / REFERENCE_RUN_ID
PROFILES_V2_DIR = Path.home() / ".sim_bench" / "profiles_v2"

# Numbers from the reference run (verified 2026-05-30 against the actual DB).
# Any new tab spec that changes these must update README.md too.
EXPECTED_N_CLUSTERS = 15
EXPECTED_N_FACES_TOTAL = 340
EXPECTED_N_FACES_ASSIGNED = 107      # not in noise
EXPECTED_BIGGEST_CLUSTER_SIZE = 35   # cluster_1
EXPECTED_CLUSTER_0_SIZE = 24
EXPECTED_CLUSTER_0_FIRST_FACE_ID = 0  # first face_id (sorted) in cluster 0

PORT = 8889
APP_URL = f"http://localhost:{PORT}"
APP_ENTRY = (
    Path(__file__).resolve().parents[3] / "app" / "face_clustering_v2" / "main.py"
)

PIPELINE_TIMEOUT_S = 600
PAGE_TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            return False


@pytest.fixture(scope="session")
def streamlit_server() -> Iterator[str]:
    """Spawn `streamlit run app/face_clustering_v2/main.py` on port 8889.
    Tears down at session exit. Skips when the app entry is missing."""
    if not APP_ENTRY.exists():
        pytest.skip(f"App entry not found: {APP_ENTRY}")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_ENTRY),
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "STREAMLIT_SERVER_RUN_ON_SAVE": "false"},
    )
    try:
        for _ in range(60):
            if _port_open(PORT):
                break
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                pytest.fail(f"Streamlit exited during startup:\n{out[:2000]}")
            time.sleep(0.5)
        else:
            pytest.fail(f"Streamlit didn't open port {PORT} within 30 s.")
        yield APP_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def page(streamlit_server):
    """Chromium page pointed at the running app. Saves screenshot on failure."""
    pw_sync = pytest.importorskip("playwright.sync_api")
    with pw_sync.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        page.goto(streamlit_server)
        page.wait_for_selector("h1", state="visible", timeout=PAGE_TIMEOUT_MS)
        try:
            yield page
        finally:
            try:
                artifact_dir = Path(__file__).parent / "_failure_artifacts"
                artifact_dir.mkdir(exist_ok=True)
                page.screenshot(path=str(artifact_dir / f"{int(time.time())}.png"))
            except Exception:
                pass
            browser.close()
