"""spec-115 T4.6 + spec-116 R15 - the Jobs page shows and cancels jobs; it no longer starts them (Playwright).

Isolated: its own control DB, job store and feature cache in a temp dir; photos are generated (no private data).
Jobs are made through the API (the regression scripts' path); the app runs them locally, ``minimal`` only.
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
                 "ALBUMIFY_FEATURE_CACHE": str(tmp / "cache.db"), "SIM_BENCH_DB": str(tmp / "sim_bench.db"),
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


def _submit(env, folder: Path) -> str:
    r = requests.post(f"{env['api']}/jobs", json={"source": str(folder), "runner": "local", "pipeline": "minimal"})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def _finished_job(env) -> str:
    """A finished local job made through the API, so tests do not depend on each other's order."""
    job_id = _submit(env, env["trip"])
    deadline = time.time() + JOB_TIMEOUT_MS / 1000
    while requests.get(f"{env['api']}/jobs/{job_id}").json()["status"] not in ("succeeded", "failed"):
        assert time.time() < deadline
        time.sleep(2)
    return job_id


def _open_job(page, job_id: str) -> None:
    page.reload()
    page.get_by_role("button", name="Jobs").click()
    page.get_by_role("heading", name="Jobs").wait_for(timeout=30_000)
    page.get_by_role("combobox", name="Job").click()
    page.get_by_role("option", name=job_id).first.click()


def test_no_run_form(page, env):
    """R15: the form, the runner choice and the re-run tools are gone; the history stays."""
    job_id = _finished_job(env)
    _open_job(page, job_id)
    page.get_by_role("tab", name="Picks").wait_for(timeout=60_000)
    page.screenshot(path=str(SHOTS / "1_album.png"), full_page=True)
    for gone in ("Run an album", "Photo folder on this PC", "Re-run these photos"):
        assert page.get_by_text(gone).count() == 0
    page.get_by_text("Job history and settings").click()
    page.get_by_text("staged 8 photos", exact=False).first.wait_for(timeout=10_000)


def test_a_script_job_says_it_has_no_album_row(page, env):
    _open_job(page, _finished_job(env))
    page.get_by_text("it has no row in Albums").wait_for(timeout=30_000)


def test_a_run_links_to_albums(page, env):
    """R15: each job of a run links to it in Albums."""
    album = requests.post(f"{env['api']}/api/v1/albums/", json={"name": "Linked", "source_path": str(env["trip"])}).json()
    steps = requests.get(f"{env['api']}/api/v1/config/pipelines").json()["minimal_pipeline"]
    run_id = requests.post(f"{env['api']}/api/v1/pipeline/run", json={
        "album_id": album["id"], "steps": steps, "pipeline_name": "minimal_pipeline"}).json()["job_id"]
    deadline = time.time() + JOB_TIMEOUT_MS / 1000
    while requests.get(f"{env['api']}/api/v1/pipeline/{run_id}").json()["status"] != "completed":
        assert time.time() < deadline
        time.sleep(2)
    _open_job(page, run_id)
    page.get_by_role("button", name="Open this run in Albums").click(timeout=30_000)
    page.get_by_role("heading", name="Runs").wait_for(timeout=30_000)
    page.locator("strong", has_text="Linked").wait_for(timeout=10_000)


def test_cancel_a_running_job(page, env):
    """spec-115 R4: Cancel on the live status ends the job failed / cancelled by owner."""
    job_id = _submit(env, env["big"])
    _open_job(page, job_id)
    page.get_by_role("button", name="Cancel this job").click(timeout=60_000)
    page.get_by_text("cancelled by owner").first.wait_for(timeout=60_000)
    page.screenshot(path=str(SHOTS / "2_cancelled.png"))
    job = requests.get(f"{env['api']}/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["error"] == "cancelled by owner"
