"""Screenshot the Merged Clusters detail panel + count loaded crop <img>s."""
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8901
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
    url = (f"http://localhost:{PORT}/?current_run_dir={urllib.parse.quote(str(REF))}"
           f"&selected_merge_pair=0,11&page=Merged%20Clusters")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": 1500, "height": 1400}).new_page()
        pg.set_default_timeout(60000)
        pg.goto(url)
        pg.wait_for_selector(r"text=/Pair \(cluster_a=/", timeout=60000)
        pg.wait_for_selector("text=/cluster_b =/", timeout=30000)
        print("img:visible immediately:", pg.locator("img:visible").count())
        # wait for the media endpoint to actually load the crops
        try:
            pg.wait_for_function("document.querySelectorAll('img').length > 0 && "
                                 "[...document.querySelectorAll('img')].some(i=>i.naturalWidth>0)",
                                 timeout=20000)
        except Exception as e:
            print("wait_for_function timed out:", e)
        pg.wait_for_timeout(1500)
        print("img:visible after wait:", pg.locator("img:visible").count())
        print("img total:", pg.locator("img").count())
        pg.screenshot(path=str(OUT / "SHOT_merged_detail.png"), full_page=True)
        print("saved SHOT_merged_detail.png")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
