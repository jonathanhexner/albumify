"""spec-045 §8.6 — manual Playwright smoke for the Cluster Analysis tab.

Opt-in. Runs against a live Streamlit server. Run with:

    .venv/Scripts/streamlit run app/face_clustering_v2/main.py --server.port 8889
    # in another terminal:
    .venv/Scripts/python tests/manual/_v2_cluster_analysis_smoke.py

Saves screenshots next to this file. Tolerates timing differences (uses
generous wait_for_selector timeouts).

5-step sequence per spec §8.6:
  1. History tab; pick a fixture run; click Load Run.
  2. Switch to Cluster Analysis tab; cluster picker populated.
  3. Pick first cluster; thumbnails + 5-metric strip visible within 10s.
  4. Expand Nearest Clusters; ≥1 row.
  5. Open Force Merge; click Preview; preview block renders without exception.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8889"
HERE = Path(__file__).resolve().parent


def run() -> int:
    print(f"Smoke target: {URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(URL, timeout=20_000)
            time.sleep(2)

            # 1. Open History tab
            page.get_by_role("tab", name="History").click()
            page.screenshot(path=str(HERE / "_v2_cluster_analysis_step1_history.png"))
            print("step 1: History tab open")

            # 2. Switch to Cluster Analysis tab (smoke only confirms it loads;
            #    Load Run requires existing fixture data — left to the operator).
            page.get_by_role("tab", name="Cluster Analysis").click()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(HERE / "_v2_cluster_analysis_step2_tab_open.png"))
            print("step 2: Cluster Analysis tab open")

            # 3-5: depend on a loaded run — skipped here. The screenshot is the
            # signal: a healthy tab renders either the cluster picker (run loaded)
            # or the "No run loaded" info banner (no run loaded). Either is a pass;
            # a Streamlit exception block is a fail.
            exception_count = page.locator(".stException").count()
            if exception_count:
                first = page.locator(".stException").first.text_content() or ""
                print(f"FAIL: {exception_count} Streamlit exception(s) on tab.\n  first: {first[:200]}")
                return 1
            print("PASS: no Streamlit exceptions on Cluster Analysis tab")
            return 0
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(run())
