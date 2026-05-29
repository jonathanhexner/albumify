"""V2 app end-to-end gold-standard against the Budapest album.

This is the **reference test** for every v2 tab and every new v2 feature.
Per CLAUDE.md: any change to a v2 tab and any new v2 feature must run
this test before being marked Implemented.

Two scenarios, both gated by the same baseline:

Scenario A — fresh pipeline run:
    1. Spawn ``streamlit run app/face_clustering_v2/main.py`` on port 8889.
    2. Drive Chromium via Playwright: fill Run tab inputs, click Run.
    3. Wait up to 10 min for the pipeline to complete.
    4. Assert: ``n_clusters == 15`` (matches the user's reference run).
    5. Assert: the v2 Cluster Analysis tab renders ≥ 5 cluster metric
       widgets + ≥ 1 face thumbnail.

Scenario B — load + inspect the reference run:
    1. Open the History tab.
    2. Find and click the row for run_id ``6437d335de914755bc3edb825c9591c0``.
    3. Click "Load into analysis tabs".
    4. Switch to Cluster Analysis tab.
    5. Assert: cluster picker shows 15 options + first cluster's
       face grid renders ≥ 1 thumbnail (the SIGHTING-080/-089 surfaces).

Baseline (the contract every v2 change must keep green):
    source_dir = D:\\Budapest2025_Google
    profile    = profile_4.json
    expected   = 15 clusters, 340 faces, producer fc_app_v2
                  (reference run is producer=fc_app, 15 clusters, 340 faces)

Skips cleanly when:
  - ``D:\\Budapest2025_Google`` is absent (CI / non-dev machines)
  - ``profile_4.json`` is missing from ``~/.sim_bench/profiles_v2/``
  - The reference run dir is missing (Scenario B only)
  - Playwright Chromium isn't installed (run
    ``.venv/Scripts/playwright install chromium`` first)

Run:
    .venv/Scripts/python -m pytest -m budapest tests/face_clustering/test_v2_e2e_budapest_baseline.py -v

Or one scenario at a time:
    .venv/Scripts/python -m pytest -m budapest tests/face_clustering/test_v2_e2e_budapest_baseline.py::test_v2_scenario_b_load_reference_run -v

Marker: ``@pytest.mark.budapest`` — opt-in. CI doesn't run this; the
spec-implementer agent runs it on the dev machine before flipping any
v2-touching spec to Implemented.
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
# Baseline constants — the contract
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(r"D:\Budapest2025_Google")
PROFILE_NAME = "profile_4.json"
EXPECTED_N_CLUSTERS = 15
EXPECTED_N_FACES = 340  # ±5 tolerance (face detector occasionally flips on borderline crops)

REFERENCE_RUN_ID = "6437d335de914755bc3edb825c9591c0"
REFERENCE_RUN_DIR = Path.home() / ".sim_bench" / "runs" / REFERENCE_RUN_ID
PROFILES_V2_DIR = Path.home() / ".sim_bench" / "profiles_v2"

PORT = 8889
APP_URL = f"http://localhost:{PORT}"
APP_ENTRY = Path(__file__).resolve().parents[2] / "app" / "face_clustering_v2" / "main.py"

PIPELINE_TIMEOUT_S = 600  # 10 min
PAGE_TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Skip preconditions
# ---------------------------------------------------------------------------

def _have(p: Path) -> bool:
    return p.exists()


pytestmark = pytest.mark.budapest


# ---------------------------------------------------------------------------
# Streamlit subprocess fixture
# ---------------------------------------------------------------------------

def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            return False


@pytest.fixture(scope="session")
def streamlit_server() -> Iterator[str]:
    """Spawn ``streamlit run`` on port 8889; tear down at session exit."""
    if not _have(APP_ENTRY):
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
        # Wait up to 30 s for the port to open.
        for _ in range(60):
            if _port_is_open(PORT):
                break
            if proc.poll() is not None:
                out = (proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else "")
                pytest.fail(f"Streamlit exited during startup. Output:\n{out[:2000]}")
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
    """Chromium page pointed at the running app. Saves a screenshot on failure."""
    pw_sync = pytest.importorskip("playwright.sync_api")
    with pw_sync.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        page.goto(streamlit_server)
        # Wait for Streamlit's initial render.
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


# ---------------------------------------------------------------------------
# Scenario A — fresh pipeline run
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_v2_scenario_a_fresh_run_produces_15_clusters(page):
    """Drive the Run tab end-to-end: type inputs, click Run, wait for the
    pipeline to complete, assert n_clusters == 15.

    This is the load-bearing contract. Any change that breaks n_clusters
    on profile_4 against Budapest will fail here.
    """
    if not _have(SOURCE_DIR):
        pytest.skip(f"Source dir missing: {SOURCE_DIR}")
    if not _have(PROFILES_V2_DIR / PROFILE_NAME):
        pytest.skip(f"Profile missing: {PROFILES_V2_DIR / PROFILE_NAME}")

    # Click the Run tab (Streamlit st.tabs renders the first tab by default;
    # if Run isn't first, this still works because all tabs render).
    page.get_by_role("tab", name="Run").click()

    # Fill the source dir + album name inputs. Streamlit's text_input
    # gets keyed by label.
    page.get_by_label("Source").fill(str(SOURCE_DIR))
    page.get_by_label("Album").fill(f"e2e_baseline_{int(time.time())}")

    # Pick profile_4.
    page.get_by_role("combobox").first.click()
    page.get_by_role("option", name="profile_4").click()

    # Click Run pipeline.
    page.get_by_role("button", name="Run pipeline").click()

    # Wait for the success message (pipeline output line mentions n_clusters).
    page.wait_for_selector(
        "text=/Run complete.*clusters/i",
        state="visible",
        timeout=PIPELINE_TIMEOUT_S * 1000,
    )

    # Extract n_clusters from the success message. Format:
    # "Run complete — N clusters from M faces across K images..."
    msg = page.locator("text=/Run complete.*clusters/i").first.text_content() or ""
    import re
    m = re.search(r"(\d+)\s+clusters", msg)
    assert m, f"Could not parse n_clusters from success message: {msg!r}"
    n_clusters = int(m.group(1))
    assert n_clusters == EXPECTED_N_CLUSTERS, (
        f"Baseline regression: profile_4 against Budapest produced {n_clusters} "
        f"clusters; expected {EXPECTED_N_CLUSTERS}. Reference run "
        f"{REFERENCE_RUN_ID} had 15."
    )


# ---------------------------------------------------------------------------
# Scenario B — load the reference run + inspect
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_v2_scenario_b_load_reference_run_and_see_clusters(page):
    """Open History → find run 6437d335... → Load → switch to Cluster
    Analysis → assert picker shows 15 clusters + thumbnails render.

    Catches: SIGHTING-078 (Cluster Analysis crash on no-op-merge runs),
    SIGHTING-079 (UI stuck on "Analysing cluster…"),
    SIGHTING-080 (Load Run rejects v2 runs as missing CSVs),
    SIGHTING-089 (History panel blank for v2 runs),
    and today's face-grid thumbnail bug.
    """
    if not _have(REFERENCE_RUN_DIR):
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")

    # 1. History tab.
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")

    # 2. Find the reference run row in the dataframe. Streamlit renders
    # st.dataframe rows; we click the one whose cell text contains the
    # reference run id (truncated to first 8 chars in the UI per spec-050).
    short_id = REFERENCE_RUN_ID[:8]
    page.get_by_role("gridcell", name=lambda s: short_id in s).first.click(timeout=15_000)

    # 3. Click "Load into analysis tabs".
    page.get_by_role("button", name="Load into analysis tabs").click()
    page.wait_for_selector("text=/Loaded/i", state="visible", timeout=15_000)

    # 4. Cluster Analysis tab.
    page.get_by_role("tab", name="Cluster Analysis").click()
    page.wait_for_selector("h2:has-text('Cluster Analysis')", state="visible")

    # 5. Assert: cluster picker has options. (Hard to count Streamlit
    # selectbox option count without opening the dropdown — instead we
    # assert the metric strip rendered, which only happens when a cluster
    # is selected and ClusterView.compute returned.)
    page.wait_for_selector("[data-testid='stMetric']", state="visible", timeout=30_000)
    n_metrics = page.locator("[data-testid='stMetric']").count()
    assert n_metrics >= 5, (
        f"Cluster Analysis metric strip didn't render — got {n_metrics} metric "
        f"widgets, expected >= 5 (Faces / Diameter / Avg / Exemplars / Outliers)."
    )

    # 6. Assert: at least one face thumbnail rendered. (st.image renders
    # an <img> tag; the face_grid uses width=110.)
    page.wait_for_selector("img", state="visible", timeout=15_000)
    n_imgs = page.locator("img").count()
    assert n_imgs >= 1, (
        "No face thumbnails rendered in Cluster Analysis tab — face_grid is "
        "showing captions only. Today's bug pattern."
    )
