"""Screenshot spec-083: Images grid, image-analysis detail (boxes), FM grid."""
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8905
REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"
OUT = Path(__file__).parent

log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app/face_clustering_v2/main.py",
     "--server.port", str(PORT), "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=log, stderr=subprocess.STDOUT,
)
base = f"http://localhost:{PORT}/?current_run_dir={urllib.parse.quote(str(REF))}"
try:
    for _ in range(60):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.5)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": 1500, "height": 1300}).new_page()
        pg.set_default_timeout(60000)

        # 1. Images grid
        pg.goto(base + "&page=Images")
        pg.wait_for_selector("h2:has-text('Images')", timeout=60000)
        pg.get_by_role("button", name="Open").first.wait_for(timeout=60000)
        pg.wait_for_function("[...document.querySelectorAll('img')].some(i=>i.naturalWidth>0)", timeout=30000)
        pg.wait_for_timeout(800)
        pg.screenshot(path=str(OUT / "SHOT_images_grid.png"), full_page=False)
        print("saved SHOT_images_grid.png; Open buttons:",
              pg.get_by_role("button", name="Open").count())

        # 2. Click first image's Open -> detail with boxes
        pg.get_by_role("button", name="Open").first.click()
        pg.wait_for_selector("text=People in this photo", timeout=30000)
        pg.wait_for_selector(".js-plotly-plot", timeout=30000)
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(OUT / "SHOT_image_detail.png"), full_page=True)
        print("saved SHOT_image_detail.png; plotly:", pg.locator(".js-plotly-plot").count())

        # 3. Face Metrics grid
        pg.goto(base + "&page=Face%20Metrics")
        pg.wait_for_selector("h2:has-text('Face Metrics')", timeout=60000)
        pg.get_by_role("button", name="Open").first.wait_for(timeout=60000)
        pg.wait_for_function("[...document.querySelectorAll('img')].some(i=>i.naturalWidth>0)", timeout=30000)
        pg.wait_for_timeout(800)
        pg.screenshot(path=str(OUT / "SHOT_fm_grid.png"), full_page=False)
        print("saved SHOT_fm_grid.png; Open buttons:",
              pg.get_by_role("button", name="Open").count())
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
