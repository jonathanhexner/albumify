"""spec-041 — Playwright smoke for the new Run tab.

Headless: opens the v2 app, asserts the H1, asserts representative widgets
from every section render, captures a screenshot. Manual / opt-in — not
part of pytest collection.
"""
import sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:8889"
SHOT = "tests/manual/_v2_smoke.png"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page()
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1:has-text('Face Clustering')", timeout=15000)

        # Tabs render (Streamlit uses data-baseweb="tab", not role=tab).
        page.wait_for_timeout(1000)
        body_initial = page.locator("body").inner_text()
        for tab in ("Run", "Clusters"):
            assert tab in body_initial, f"tab label {tab!r} not present in body"

        # Section headers — expanders are open by default for required sections.
        expected_substrings = [
            "Stage 3 · Quality Gate",
            "Stage 5 · Cluster",
            "Stage 6 · Exemplars",
            "Optional Stages",
        ]
        body = page.locator("body").inner_text()
        for s in expected_substrings:
            assert s in body, f"expected section header {s!r} not found in body"

        # Click the Stage 6 expander open (lazy-rendered when collapsed).
        try:
            page.get_by_text("Stage 6 · Exemplars", exact=False).first.click(timeout=2000)
            page.wait_for_timeout(300)
            body = page.locator("body").inner_text()
        except Exception:
            pass

        # Representative widget labels from the default-expanded sections.
        widget_labels = [
            "blur_min", "max_faces_per_image_core", "yaw_max",
            "K (kNN neighbours)", "distance_threshold", "min_cluster_size",
            "split_enabled", "merge_enabled", "attach_enabled",
        ]
        for w in widget_labels:
            assert w in body, f"widget label {w!r} not present"

        # Merge sub-panel is gated by merge_enabled (default True); confirm it's rendered.
        assert "Merge Parameters" in body, "merge sub-panel not rendered when merge_enabled=True"
        assert "merge_candidate_threshold" in body, "merge widget missing"

        page.screenshot(path=SHOT, full_page=True)
        b.close()
    print(f"OK — screenshot: {SHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
