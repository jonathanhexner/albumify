"""Probe: does clicking a Face Metrics row actually navigate to Face Analysis?
Tries clicking (a) a data cell and (b) the row-select checkbox column, and
reports whether the view switches + whether the large face image appears.
"""
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8903
REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"
OUT = Path(__file__).parent

log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app/face_clustering_v2/main.py",
     "--server.port", str(PORT), "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=log, stderr=subprocess.STDOUT,
)


def radio_value(pg):
    return pg.locator("[role=radiogroup] label").filter(has=pg.locator("input:checked")).inner_text()


try:
    for _ in range(60):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.5)
    url = (f"http://localhost:{PORT}/?current_run_dir={urllib.parse.quote(str(REF))}"
           f"&page=Face%20Metrics")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": 1500, "height": 1200}).new_page()
        pg.set_default_timeout(60000)
        pg.goto(url)
        pg.wait_for_selector("h2:has-text('Face Metrics')", timeout=60000)
        pg.wait_for_selector("[data-testid='stDataFrame'] canvas", timeout=60000)
        pg.wait_for_timeout(2500)
        grid = pg.locator("[data-testid='stDataFrame'] canvas").first
        box = grid.bounding_box()

        # (a) click a DATA cell (the 'id' column area, ~x+260) on the first row
        pg.mouse.click(box["x"] + 260, box["y"] + 55)
        pg.wait_for_timeout(2500)
        print("after data-cell click -> active radio:", radio_value(pg))
        pg.screenshot(path=str(OUT / "PROBE_fm_datacell.png"), full_page=True)

        # reload + click the CHECKBOX column (~x+22)
        pg.goto(url)
        pg.wait_for_selector("[data-testid='stDataFrame'] canvas", timeout=60000)
        pg.wait_for_timeout(2500)
        box = pg.locator("[data-testid='stDataFrame'] canvas").first.bounding_box()
        pg.mouse.click(box["x"] + 22, box["y"] + 55)
        pg.wait_for_timeout(2500)
        print("after checkbox click -> active radio:", radio_value(pg))
        pg.screenshot(path=str(OUT / "PROBE_fm_checkbox.png"), full_page=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
