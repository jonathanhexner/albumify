"""spec-116 R1 R6 R7 R8 - Configure & Run chooses the machine (real app + real API + faked Modal, Playwright).

Isolated: its own sim_bench.db, control DB, job store and feature cache in a temp dir; photos are generated. The
Modal side is tests/e2e_fake_modal_api.py: no network, no spend.
Run: .venv/Scripts/python -m pytest -m e2e tests/test_configure_run_on_e2e.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

from tests.test_jobs_page_run_e2e import _free_port, _photos, _wait

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[1]
SHOTS = Path(tempfile.gettempdir()) / "albumify_configure_e2e"
RUN_TIMEOUT_S = 300


class Stack:
    def __init__(self, tmp: Path, **env):
        self.tmp, self.api_port, self.app_port = tmp, _free_port(), _free_port()
        child_env = {**os.environ, "SIM_BENCH_DB": str(tmp / "sim_bench.db"),
                     "DATABASE_URL": f"sqlite:///{(tmp / 'control.db').as_posix()}",
                     "ALBUMIFY_JOBS_ROOT": str(tmp / "jobs"), "ALBUMIFY_FEATURE_CACHE": str(tmp / "cache.db"),
                     "API_BASE_URL": f"http://localhost:{self.api_port}", **env}
        self.api_url, self.app_url = f"http://localhost:{self.api_port}", f"http://localhost:{self.app_port}"
        self.procs = [
            subprocess.Popen([sys.executable, "tests/e2e_fake_modal_api.py", str(self.api_port)], cwd=REPO,
                             env={**child_env, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL),
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit/main.py", "--server.port",
                              str(self.app_port), "--server.headless", "true"], cwd=REPO, env=child_env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
        _wait(f"{self.api_url}/jobs")
        _wait(self.app_url)

    def album(self, name: str, folder: Path) -> dict:
        r = requests.post(f"{self.api_url}/api/v1/albums/", json={"name": name, "source_path": str(folder)})
        assert r.status_code < 300, r.text
        return r.json()

    def stop(self):
        for p in self.procs:
            p.terminate()
            p.wait(timeout=30)


@pytest.fixture(scope="module")
def stack():
    tmp = Path(tempfile.mkdtemp(prefix="albumify_e2e_"))
    _photos(tmp / "trip", 8)
    _photos(tmp / "big", 60)        # long enough that Cancel lands while it runs
    s = Stack(tmp)
    s.album("Trip", tmp / "trip")
    s.album("Big", tmp / "big")
    yield s
    s.stop()


@pytest.fixture
def page(stack):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1500, "height": 1200})
        pg.goto(stack.app_url)
        yield pg
        browser.close()


def open_configure(page, album: str, pipeline: str = "Minimal (no faces)") -> None:
    page.get_by_role("button", name="Configure").click()
    page.get_by_role("heading", name="Configure & Run").wait_for(timeout=30_000)
    page.get_by_role("combobox").first.click()
    page.get_by_role("option", name=album).first.click()
    page.wait_for_timeout(1000)
    page.get_by_text(pipeline).click()
    page.wait_for_timeout(1000)


def run_on_l4(page) -> None:
    page.get_by_text("L4 (GPU)").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Run Pipeline").click()
    page.get_by_text("This runs on the L4:").wait_for(timeout=30_000)


def runs(stack) -> list:
    return requests.get(f"{stack.api_url}/api/v1/pipeline/runs").json()


def finished(stack, run_id: str) -> dict:
    end = time.time() + RUN_TIMEOUT_S
    while time.time() < end:
        row = next(r for r in runs(stack) if r["run_id"] == run_id)
        if row["status"] in ("completed", "failed"):
            return row
        time.sleep(1)
    pytest.fail(f"run {run_id} did not finish")


def test_l4_needs_confirm(page, stack):
    """R1: the L4 shows its estimate and waits for a second click; Back leaves nothing behind."""
    before = len(runs(stack))
    open_configure(page, "Trip")
    run_on_l4(page)
    page.get_by_text("8 photos").first.wait_for(timeout=5_000)
    page.screenshot(path=str(SHOTS / "1_l4_confirm.png"), full_page=True)
    page.get_by_role("button", name="Back").click()
    page.wait_for_timeout(1500)
    assert len(runs(stack)) == before
    assert page.get_by_role("button", name="Run Pipeline").is_visible()


def test_l4_run_opens_in_results(page, stack):
    """R6: confirm, watch the stages, then the run is in Albums and opens in Results on the original photos."""
    open_configure(page, "Trip")
    run_on_l4(page)
    page.get_by_role("button", name="Confirm: run on the L4").click()
    page.get_by_text("on L4 (GPU)").first.wait_for(timeout=30_000)
    page.screenshot(path=str(SHOTS / "2_l4_running.png"), full_page=True)
    row = finished(stack, runs(stack)[0]["run_id"])
    assert row["status"] == "completed" and row["ran_on"] == "L4 (GPU)" and row["photos"] == 8

    images = requests.get(f"{stack.api_url}/api/v1/results/{row['run_id']}/images").json()
    assert len(images) == 8
    assert all(Path(i["path"]).parent.samefile(stack.tmp / "trip") for i in images), images[0]["path"]
    assert all(Path(i["path"]).is_file() for i in images)

    page.get_by_role("button", name="Albums").click()
    page.get_by_role("heading", name="Runs").wait_for(timeout=30_000)
    page.get_by_role("button", name="Open").first.click()
    page.get_by_text("Run", exact=True).first.wait_for(timeout=30_000)
    page.wait_for_timeout(3000)
    page.screenshot(path=str(SHOTS / "3_l4_results.png"), full_page=True)
    assert page.get_by_text(row["run_id"][:8]).first.is_visible()
    assert page.locator('[data-testid="stException"]').count() == 0


def test_progress_and_cancel(page, stack):
    """R7: steps on this PC; Cancel ends the run failed, with the message on Configure & Run and no rows."""
    open_configure(page, "Big")
    page.get_by_role("button", name="Run Pipeline").click()
    page.get_by_text("Pipeline Running...").wait_for(timeout=30_000)
    page.get_by_text("on This PC").first.wait_for(timeout=60_000)
    page.screenshot(path=str(SHOTS / "4_local_running.png"), full_page=True)
    page.get_by_role("button", name="Cancel Pipeline").click()
    page.get_by_text("cancelled by owner").first.wait_for(timeout=60_000)
    page.screenshot(path=str(SHOTS / "5_cancelled.png"), full_page=True)
    row = runs(stack)[0]
    assert row["status"] == "failed" and row["photos"] is None
    assert requests.get(f"{stack.api_url}/jobs/{row['run_id']}").json()["error"] == "cancelled by owner"


def test_failure_message_on_configure(tmp_path):
    """R8: Modal unreachable ends the run failed, with the reason shown where it was started."""
    from playwright.sync_api import sync_playwright

    from tests.test_jobs_page_run_e2e import _photos as make
    make(tmp_path / "trip", 2)
    s = Stack(tmp_path, FAKE_MODAL_FAIL="1")
    try:
        s.album("Trip", tmp_path / "trip")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            pg = browser.new_page(viewport={"width": 1500, "height": 1200})
            pg.goto(s.app_url)
            open_configure(pg, "Trip")
            run_on_l4(pg)
            pg.get_by_role("button", name="Confirm: run on the L4").click()
            pg.get_by_text("Modal call failed: unreachable").first.wait_for(timeout=60_000)
            browser.close()
        assert runs(s)[0]["status"] == "failed"
    finally:
        s.stop()
