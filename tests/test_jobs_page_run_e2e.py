"""spec-115 T4.6 - run, watch, cancel and re-run albums from the Jobs page (real API + real app, Playwright).

Isolated: its own control DB, job store and feature cache in a temp dir; photos are generated (no private data).
Runs local ``minimal`` jobs only; the L4 path is checked up to its confirm step and backed out of (no spend).
Run: .venv/Scripts/python -m pytest -m e2e tests/test_jobs_page_run_e2e.py -v
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[1]
SHOTS = Path(tempfile.gettempdir()) / "albumify_jobs_run_e2e"
JOB_TIMEOUT_MS = 300_000


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


def _photos(folder: Path, n: int) -> None:
    rng = np.random.default_rng(7)
    folder.mkdir(parents=True)
    for i in range(n):
        base = rng.integers(0, 255, size=3)
        img = np.clip(rng.normal(base, 40, size=(480, 640, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(folder / f"p{i:02d}.jpg", quality=92)


@pytest.fixture(scope="module")
def env():
    tmp = Path(tempfile.mkdtemp(prefix="albumify_e2e_"))
    _photos(tmp / "trip", 8)
    _photos(tmp / "big", 60)          # long enough that Cancel lands while it runs
    api_port, app_port = _free_port(), _free_port()
    child_env = {**os.environ,
                 "DATABASE_URL": f"sqlite:///{(tmp / 'control.db').as_posix()}",
                 "ALBUMIFY_JOBS_ROOT": str(tmp / "jobs"),
                 "ALBUMIFY_FEATURE_CACHE": str(tmp / "cache.db"),
                 "API_BASE_URL": f"http://localhost:{api_port}"}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "sim_bench.api.main:app", "--port", str(api_port)],
                           cwd=REPO, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit/main.py",
                            "--server.port", str(app_port), "--server.headless", "true"], cwd=REPO,
                           env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(f"http://localhost:{api_port}/jobs")
        _wait(f"http://localhost:{app_port}")
        yield {"app": f"http://localhost:{app_port}", "api": f"http://localhost:{api_port}", "trip": tmp / "trip",
               "big": tmp / "big"}
    finally:
        for p in (api, app):
            p.terminate()
            p.wait(timeout=30)


@pytest.fixture
def page(env):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(env["app"])
        pg.get_by_role("button", name="Jobs").click()
        pg.get_by_role("heading", name="Jobs").wait_for(timeout=30_000)
        yield pg
        browser.close()


def _open_run_form(page) -> None:
    box = page.get_by_label("Photo folder on this PC")
    if box.count() == 0 or not box.is_visible():
        page.get_by_text("Run an album").click()
        box.wait_for(state="visible", timeout=10_000)


def _choose(page, label: str, option: str, nth: int = 0) -> None:
    page.get_by_role("combobox", name=label).nth(nth).click()
    page.get_by_role("option", name=option, exact=True).click()
    page.wait_for_timeout(500)


def _start_minimal(page, folder: Path, n: int | None = None) -> None:
    _open_run_form(page)
    box = page.get_by_label("Photo folder on this PC")
    box.fill(str(folder))
    box.press("Enter")
    if n:
        how_many = page.get_by_label("How many (0 = all)")
        how_many.fill(str(n))
        how_many.press("Enter")
    _choose(page, "Pipeline", "minimal")
    page.wait_for_timeout(800)
    page.screenshot(path=str(SHOTS / "0_before_start.png"), full_page=True)
    page.get_by_role("button", name="Start", exact=True).first.click()


def test_run_from_form_status_updates_without_reload_and_album_opens(page, env):
    """R1 + R3: start a local job from the form; the status moves on its own; the album opens."""
    _start_minimal(page, env["trip"])
    page.get_by_text("on This PC").first.wait_for(timeout=60_000)        # live status block
    page.screenshot(path=str(SHOTS / "1_running.png"))
    page.get_by_role("tab", name="Picks").wait_for(timeout=JOB_TIMEOUT_MS)  # no reload by the test
    page.screenshot(path=str(SHOTS / "2_album.png"), full_page=True)
    jobs = requests.get(f"{env['api']}/jobs").json()
    assert jobs[0]["status"] == "succeeded"
    detail = requests.get(f"{env['api']}/jobs/{jobs[0]['id']}").json()
    assert detail["settings"]["pipeline"] == "minimal" and detail["settings"]["runner"] == "local"
    assert any(e["message"].startswith("staged 8 photos") for e in detail["events"])


def test_l4_needs_confirm(page, env):
    """R2: the L4 shows its estimate and waits for a second click; Back leaves no job behind."""
    before = len(requests.get(f"{env['api']}/jobs").json())
    _open_run_form(page)
    box = page.get_by_label("Photo folder on this PC")
    box.fill(str(env["trip"]))
    box.press("Enter")
    _choose(page, "Run on", "Modal L4 (GPU)")
    page.get_by_role("button", name="Start", exact=True).first.click()
    page.get_by_text("This runs on the Modal L4: 8 photos").wait_for(timeout=30_000)
    page.screenshot(path=str(SHOTS / "3_l4_confirm.png"))
    page.get_by_role("button", name="Back").click()
    page.wait_for_timeout(1500)
    assert len(requests.get(f"{env['api']}/jobs").json()) == before


def _finished_job(env) -> str:
    """A finished local job made through the API, so tests do not depend on each other's order."""
    r = requests.post(f"{env['api']}/jobs", json={"source": str(env["trip"]), "runner": "local",
                                                    "pipeline": "minimal"})
    job_id = r.json()["job_id"]
    deadline = time.time() + JOB_TIMEOUT_MS / 1000
    while requests.get(f"{env['api']}/jobs/{job_id}").json()["status"] not in ("succeeded", "failed"):
        assert time.time() < deadline
        time.sleep(2)
    return job_id


def test_cancel_a_running_job(page, env):
    """R4: Cancel on the live status ends the job failed / cancelled by owner."""
    _start_minimal(page, env["big"])
    page.get_by_role("button", name="Cancel this job").click(timeout=60_000)
    page.get_by_text("cancelled by owner").first.wait_for(timeout=60_000)
    page.screenshot(path=str(SHOTS / "4_cancelled.png"))
    job = requests.get(f"{env['api']}/jobs").json()[0]
    assert job["status"] == "failed" and job["error"] == "cancelled by owner"


def test_rerun_opens_compare(page, env):
    """R5 + R12: re-run a finished job's photos; Compare runs opens on the pair, with its settings section."""
    first = _finished_job(env)
    page.reload()                                   # a new session starts on Home
    page.get_by_role("button", name="Jobs").click()
    page.get_by_role("heading", name="Jobs").wait_for(timeout=30_000)
    page.get_by_role("combobox", name="Job").click()
    page.get_by_role("option", name=f"{first}  -  succeeded").first.click()
    page.get_by_role("tab", name="Picks").wait_for(timeout=60_000)
    page.get_by_text("Re-run these photos").click()
    _choose(page, "Pipeline", "minimal", nth=0)
    page.get_by_role("button", name="Start", exact=True).last.click()
    page.get_by_role("heading", name="Compare runs").wait_for(timeout=JOB_TIMEOUT_MS)
    page.get_by_text("Settings that differ").wait_for(timeout=30_000)
    page.screenshot(path=str(SHOTS / "5_compare.png"), full_page=True)
    jobs = requests.get(f"{env['api']}/jobs").json()
    newest = jobs[0]
    assert newest["status"] == "succeeded" and newest["id"] != first
    events = requests.get(f"{env['api']}/jobs/{newest['id']}").json()["events"]
    assert any(e["message"] == f"re-run of {first}" for e in events)
