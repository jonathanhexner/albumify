"""spec-042 H4 — Playwright smoke for the v2 History tab.

Opt-in (not in default pytest discovery). Requires a running Streamlit
on port 8889. Asserts:

* The History tab is present.
* Switching to it renders the filter bar (Album / Date range / Search).
* The "Pipeline Runs" subheader appears.
* No Streamlit exception markdown is on the page.

The intent is to catch the same class of failure that the spec-041
yaw_max=999 case slipped through: a layout / widget binding regression
that wouldn't fail unit tests because it only manifests when Streamlit
actually renders.
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    """Run the smoke. Returns 0 on success, raises AssertionError on failure."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8889", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1:has-text('Face Clustering')", timeout=15000)

        body_initial = page.locator("body").inner_text()
        assert "History" in body_initial, "History tab label not present in body"

        # Click the History tab.
        page.get_by_text("History", exact=True).first.click()
        page.wait_for_timeout(2000)  # rerun + render

        body = page.locator("body").inner_text()

        # Section header visible.
        assert "Pipeline Runs" in body, (
            f"Pipeline Runs heading missing after History click; body head: {body[:300]!r}"
        )

        # Filter bar widgets visible.
        for label in ("Album", "Date range", "Search"):
            assert label in body, f"filter widget label {label!r} missing"

        # No Streamlit error markdown.
        for marker in (
            "StreamlitAPIException",
            "StreamlitValueAboveMaxError",
            "StreamlitValueBelowMinError",
            "ValidationError",
            "Traceback",
        ):
            assert marker not in body, (
                f"Streamlit error appeared in History tab: {marker}"
            )

        page.screenshot(
            path="tests/manual/_v2_history_smoke.png", full_page=True
        )
        browser.close()
    print("OK - History tab renders without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
