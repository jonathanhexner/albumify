"""spec-114 T4.1 - Compare runs page, real API + real app + the real regression jobs (Playwright).

Needs the 2026-10-16 regression jobs in the local job list (tags regression:<album>[:baseline]); skips
otherwise. Starts its own API and Streamlit on free ports. Screenshots stay on disk (they show photos).
Run: .venv/Scripts/python -m pytest -m "e2e and needs_data" tests/test_compare_page_e2e.py -v
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

pytestmark = [pytest.mark.e2e, pytest.mark.needs_data]

REPO = Path(__file__).resolve().parents[1]
SHOTS = Path(tempfile.gettempdir()) / "albumify_compare_e2e"
BASELINE, CHECK = "slovakia-20261016-132804", "slovakia-20261016-174613"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout_s: float = 120) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            if requests.get(url, timeout=2).status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    pytest.fail(f"{url} did not come up in {timeout_s}s")


@pytest.fixture(scope="module")
def app_url():
    api_port, app_port = _free_port(), _free_port()
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "sim_bench.api.main:app", "--port", str(api_port)],
                           cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit/main.py",
                            "--server.port", str(app_port), "--server.headless", "true"], cwd=REPO,
                           env={**os.environ, "API_BASE_URL": f"http://localhost:{api_port}"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://localhost:{api_port}/jobs")
        ids = {j["id"] for j in requests.get(f"http://localhost:{api_port}/jobs?limit=1000").json()}
        if not {BASELINE, CHECK} <= ids:
            pytest.skip("the 2026-10-16 regression jobs are not in this machine's job list")
        _wait(f"http://localhost:{app_port}")
        yield f"http://localhost:{app_port}"
    finally:
        for p in (api, app):
            p.terminate()
            p.wait(timeout=20)


@pytest.fixture
def page(app_url):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(app_url)
        pg.get_by_role("button", name="Compare runs").click()
        pg.get_by_role("heading", name="Compare runs").wait_for(timeout=30_000)
        yield pg
        browser.close()


def _settle(page) -> None:
    """Wait until Streamlit has finished the rerun a click started (else we read the old page)."""
    page.wait_for_timeout(300)
    page.locator("[data-testid=stStatusWidget]").wait_for(state="detached", timeout=60_000)


def _choose(page, label: str, option: str) -> None:
    page.locator("[data-testid=stMain]").get_by_label(label).click()
    page.get_by_role("option", name=option).first.click()
    _settle(page)


def test_regression_album_preselects_baseline_and_shows_the_verdict(page):
    _choose(page, "Regression album", "slovakia")
    page.get_by_text("Close enough").wait_for(timeout=30_000)
    body = page.inner_text("[data-testid=stMain]")
    assert BASELINE in body and "[baseline]" in body and CHECK in body
    metrics = page.locator("[data-testid=stMetric]").all_inner_texts()
    assert any("Swapped" in m and "5" in m for m in metrics), metrics
    assert any("Missing" in m and "0" in m for m in metrics), metrics
    page.screenshot(path=str(SHOTS / "verdict.png"))


def test_swapped_pair_opens_both_runs_decisions(page):
    _choose(page, "Regression album", "slovakia")
    page.get_by_role("tab", name="Swapped (5)").click()
    _settle(page)
    imgs = page.locator("[role=tabpanel]:visible img")
    imgs.first.wait_for(timeout=30_000)
    assert imgs.count() == 10
    page.get_by_role("button", name="A: 20260822_181541").click()
    _settle(page)
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout=30_000)
    text = dialog.inner_text()
    assert "20260822_181541" in text and "20260822_181544" in text
    assert "A: Kept" in text and "B: Kept" in text and "Near-duplicate" in text, text
    page.wait_for_timeout(800)                         # let the dialog finish fading in
    page.screenshot(path=str(SHOTS / "swapped_dialog.png"))


def test_people_side_by_side_with_faces(page):
    _choose(page, "Regression album", "slovakia")
    page.get_by_role("tab", name="People (79)").click()
    _settle(page)
    panel = page.locator("[role=tabpanel]:visible")
    panel.get_by_text("29").first.wait_for(timeout=30_000)
    assert "A (79 people)" in panel.inner_text() and "B (79 people)" in panel.inner_text()
    assert panel.locator("img").count() >= 20          # 15 rows x 2 runs, a few without a crop


def test_jobs_on_different_photos_are_refused_with_a_message(page):
    _choose(page, "Regression album", "Any two jobs")
    _choose(page, "A - reference", "quick-20261016-104100")
    _choose(page, "B - new run", CHECK)
    page.get_by_text("compare runs on the same photos").wait_for(timeout=30_000)


def test_jobs_page_hides_regression_runs_by_default(page):
    page.get_by_role("button", name="Jobs", exact=True).click()
    page.get_by_role("heading", name="Jobs").wait_for(timeout=30_000)
    page.get_by_label("Job").click()
    options = page.get_by_role("option").all_inner_texts()
    assert options and not any("regression:" in o for o in options), options
