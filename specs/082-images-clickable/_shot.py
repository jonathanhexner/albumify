"""Manual screenshot harness for spec-082 Images page (canvas thumbnails).
Starts the v2 app, seeds the reference run via query param, screenshots the
Images table, then clicks the first row and screenshots the image analysis.
"""
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8899
REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"
OUT = Path(__file__).parent

log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app/face_clustering_v2/main.py",
     "--server.port", str(PORT), "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=log, stderr=subprocess.STDOUT,
)
try:
    for _ in range(60):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.5)
    url = (f"http://localhost:{PORT}/?current_run_dir="
           f"{urllib.parse.quote(str(REF))}&page=Images")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": 1500, "height": 1100}).new_page()
        pg.set_default_timeout(60000)
        pg.goto(url)
        pg.wait_for_selector("h2:has-text('Images')", timeout=60000)
        pg.wait_for_selector("[data-testid='stDataFrame']", timeout=60000)
        pg.wait_for_timeout(4000)  # canvas paints thumbnails
        pg.screenshot(path=str(OUT / "SHOT_images.png"), full_page=True)
        print("saved SHOT_images.png")
        grid = pg.locator("[data-testid='stDataFrame'] canvas").first
        box = grid.bounding_box()
        if box:
            pg.mouse.click(box["x"] + 22, box["y"] + 55)  # row-select checkbox col
            pg.wait_for_timeout(4000)
            pg.screenshot(path=str(OUT / "SHOT_image_analysis.png"), full_page=True)
            print("saved SHOT_image_analysis.png")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
