"""Scenario K — Images grid: clicking an image opens its analysis in place.

spec-083: the Images tab is now a clickable thumbnail Grid (default). Clicking a
real ``st.button("Open")`` sets ``selected_image_path`` and the tab swaps to the
in-tab Image Analysis (master-detail) — the source photo with the face boxes
that passed filtration. Guards both the click wiring and the analysis render.

Click sequence:
  1. Reference run seeded by the fixture.
  2. Images tab -> wait for a grid Open button -> click it.
  3. The tab swaps to the detail view (no nav change; same tab).

Assertions:
  1. "People in this photo" caption present (detail rendered).
  2. A Plotly overlay rendered (the boxed source photo).
  3. "Back to images" returns to the grid.
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import goto_page

pytestmark = pytest.mark.budapest


def test_scenario_k_images_click(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    goto_page(page, "Images")
    page.wait_for_selector("h2:has-text('Images')", state="visible")
    open_btn = page.get_by_role("button", name="Open").first
    open_btn.wait_for(state="visible", timeout=60_000)
    open_btn.click()

    # Master-detail swap within the Images tab.
    page.wait_for_selector("text=/People in this photo/", state="visible", timeout=30_000)
    # spec-080 single-page render mounts the overlay on the rerun; wait for it.
    page.wait_for_selector(".js-plotly-plot", state="visible", timeout=30_000)
    assert page.locator(".js-plotly-plot").count() >= 1, "No boxed source photo rendered."

    page.get_by_role("button", name="Back to images").click()
    page.wait_for_selector("text=/Click .*Open.* under an image/", state="visible", timeout=15_000)
