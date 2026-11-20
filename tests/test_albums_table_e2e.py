"""spec-116 R10 - Albums is one table of runs (real app + real API, Playwright; isolated, no Modal).

Run: .venv/Scripts/python -m pytest -m e2e tests/test_albums_table_e2e.py -v
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import requests

from tests.test_configure_run_on_e2e import SHOTS, Stack
from tests.test_jobs_page_run_e2e import _photos

pytestmark = pytest.mark.e2e


def _run(stack, album_id: str) -> str:
    steps = requests.get(f"{stack.api_url}/api/v1/config/pipelines").json()["minimal_pipeline"]
    r = requests.post(f"{stack.api_url}/api/v1/pipeline/run",
                      json={"album_id": album_id, "steps": steps, "pipeline_name": "minimal_pipeline"})
    assert r.status_code == 200, r.text
    run_id = r.json()["job_id"]
    end = time.time() + 300
    while requests.get(f"{stack.api_url}/api/v1/pipeline/{run_id}").json()["status"] not in ("completed", "failed"):
        assert time.time() < end
        time.sleep(1)
    assert requests.get(f"{stack.api_url}/api/v1/pipeline/{run_id}").json()["status"] == "completed"
    return run_id


@pytest.fixture(scope="module")
def stack():
    tmp = Path(tempfile.mkdtemp(prefix="albumify_e2e_"))
    _photos(tmp / "trip", 6)
    s = Stack(tmp)
    album = s.album("Trip", tmp / "trip")
    s.first, s.second = _run(s, album["id"]), _run(s, album["id"])
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
        pg.get_by_role("button", name="Albums").click()
        pg.get_by_role("heading", name="Runs").wait_for(timeout=30_000)
        yield pg
        browser.close()


def test_table_shows_every_column(page, stack):
    for text in ("Trip", str((stack.tmp / "trip").resolve()), "This PC", "completed"):
        page.get_by_text(text, exact=True).first.wait_for(timeout=10_000)
    page.screenshot(path=str(SHOTS / "6_albums_table.png"), full_page=True)


def test_row_opens_results(page, stack):
    page.get_by_role("button", name="Open").nth(1).click()         # newest first: the second row is the older run
    page.wait_for_timeout(4000)
    page.screenshot(path=str(SHOTS / "7_opened.png"), full_page=True)
    assert page.get_by_text(stack.first[:8]).first.is_visible()
    assert page.locator('[data-testid="stException"]').count() == 0


def test_two_rows_open_compare(page, stack):
    compare = page.get_by_role("button", name="Compare runs", exact=True).last
    assert compare.is_disabled()
    boxes = page.locator('[data-testid="stCheckbox"] label')
    boxes.nth(0).click()
    page.wait_for_timeout(1000)
    assert compare.is_disabled()
    boxes.nth(1).click()
    page.wait_for_timeout(1000)
    compare.click()
    page.get_by_role("heading", name="Compare runs").wait_for(timeout=30_000)
    page.get_by_text("Settings that differ").wait_for(timeout=30_000)
    page.screenshot(path=str(SHOTS / "8_compare.png"), full_page=True)
    assert page.locator('[data-testid="stException"]').count() == 0
