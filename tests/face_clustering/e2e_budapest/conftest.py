"""Shared fixtures + baseline constants for the v2 Budapest e2e suite.

See `README.md` in this directory for the per-scenario contract:
what each test does, what it asserts, what new tabs must extend.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
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

# spec-063 Scenario C: band around the 15-cluster baseline. Reclustering the
# reference run with profile_4 defaults should land in this range; outside
# means either the clustering algorithm drifted or the recluster wiring
# corrupted the input.
EXPECTED_RECLUSTER_BAND = (12, 18)

# spec-065 Scenario E (Merged Clusters): reference run has merge_decisions
# populated; one row minimum proves the table reaches the UI.
EXPECTED_MERGE_DECISIONS_MIN_ROWS = 1

# spec-066 Scenario H (Overview): the reference run's album. Overview reads
# the global action_log, so the per-album chart must include this bar.
EXPECTED_REFERENCE_ALBUM = "Budapest2025_Google_5"

# spec-065 Scenario F (Quality): faces total - assigned = 233 rejections,
# ±7 to absorb minor gate-counting variations across pipeline iterations.
EXPECTED_REJECTED_BAND = (
    EXPECTED_N_FACES_TOTAL - EXPECTED_N_FACES_ASSIGNED - 13,  # 220
    EXPECTED_N_FACES_TOTAL - EXPECTED_N_FACES_ASSIGNED + 7,   # 240
)

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


def _wait_port_free(port: int, timeout_s: float = 15.0) -> bool:
    """Block until ``port`` is free (a prior server fully released it)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _port_open(port):
            return True
        time.sleep(0.25)
    return False


@pytest.fixture
def streamlit_server() -> Iterator[str]:
    """Spawn `streamlit run app/face_clustering_v2/main.py` on port 8889.

    Function-scoped (spec-067): a SHARED session-scoped server served only
    the first browser test reliably — the next test's ``page.goto`` timed
    out because the first session's heavy eager render (Streamlit runs all
    st.tabs bodies every script run; the reference album paints 340 face
    crops) kept the single ScriptRunner pegged. A fresh server per test
    only ever runs one session, so each scenario gets a responsive app.
    Skips when the app entry is missing.
    """
    if not APP_ENTRY.exists():
        pytest.skip(f"App entry not found: {APP_ENTRY}")

    if not _wait_port_free(PORT):
        pytest.fail(f"Port {PORT} still in use from a prior server; cannot start.")

    # IMPORTANT: redirect the child's stdout/stderr to a real file, NOT an
    # unread subprocess.PIPE. Streamlit logs to stderr on every rerun
    # (deprecation warnings, request logs, app telemetry). An un-drained PIPE
    # fills its ~64KB OS buffer after enough reruns and BLOCKS the Streamlit
    # process on its next write — the browser then hangs and every
    # wait_for_selector times out. A file sink never blocks. (Found via
    # spec-066: the new tabs added just enough per-rerun output to tip a
    # previously-passing suite over the buffer limit.)
    log_file = tempfile.NamedTemporaryFile(
        prefix="streamlit_e2e_", suffix=".log", delete=False, mode="w+b"
    )
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_ENTRY),
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "STREAMLIT_SERVER_RUN_ON_SAVE": "false"},
    )

    def _server_log() -> str:
        try:
            log_file.flush()
            return Path(log_file.name).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    try:
        for _ in range(60):
            if _port_open(PORT):
                break
            if proc.poll() is not None:
                pytest.fail(f"Streamlit exited during startup:\n{_server_log()[:2000]}")
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
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # Let the OS release the port before the next test's server spawns.
        _wait_port_free(PORT)
        try:
            log_file.close()
            os.unlink(log_file.name)
        except Exception:
            pass


@pytest.fixture
def page_with_reference_run_loaded(page):
    """``page`` with the Budapest reference run already loaded into the
    analysis tabs — without the canvas-rendered ``st.dataframe`` row-pick.

    spec-067 / SIGHTING-091: the History run-picker is glide-data-grid
    canvas, not addressable by Playwright. Instead of clicking the row +
    "Load into analysis tabs", we seed ``current_run_dir`` via query param;
    ``main.py`` mirrors it into session_state on first render, and every
    analysis tab resolves its data from that key. Equivalent end-state to
    the manual Load flow, minus the one untestable click.

    Skips when the reference run isn't on this machine.
    """
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")
    seeded = f"{APP_URL}?current_run_dir={urllib.parse.quote(str(REFERENCE_RUN_DIR))}"
    page.goto(seeded)
    page.wait_for_selector("h1", state="visible", timeout=PAGE_TIMEOUT_MS)
    return page


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
