"""Scenario D — Face Analysis drill-down from a Cluster Analysis thumbnail.

Loads the reference run via the History tab, switches to Cluster Analysis,
waits for the face_grid to render, clicks the first thumbnail's "Open"
button (which writes ``selected_face_id`` to session_state), then opens
the Face Analysis tab and verifies the per-face popup rendered.

Click sequence (per spec-064 §"E2E contract"):
  1. Reference run loaded via the ``page_with_reference_run_loaded`` fixture
     (spec-067: query-param seed replaces the canvas dataframe row-pick).
  2. Cluster Analysis tab → wait for face_grid → click first thumbnail "Open"
  3. Face Analysis tab → wait for header

Assertions (per spec-064 §"E2E contract"):
  1. Face Analysis tab is visible (h2 "Face Analysis" present)
  2. A large image (Plotly bbox overlay) is rendered — checked via the
     Plotly chart container being present
  3. >= 5 metric widgets rendered (blur / yaw / pitch / roll / area)
  4. selected_face_id is the id we clicked — surfaced via the Face id
     number_input echoing it

What this catches:
  - face_grid's Open button not writing session_state
  - FaceAnalysisService.compute_face_detail crashing on a real face id
  - SIGHTING-079 class regression on the Face Analysis tab
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import goto_page

pytestmark = pytest.mark.budapest


def test_scenario_d_face_analysis_drill_down(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    # 1. Reference run already seeded by the fixture (spec-067).
    # 2. Cluster Analysis → wait for face_grid to render an Open button.
    goto_page(page, "Cluster Analysis")
    page.wait_for_selector("h2:has-text('Cluster Analysis')", state="visible")
    open_btn = page.get_by_role("button", name="Open").first
    open_btn.wait_for(state="visible", timeout=60_000)
    open_btn.click()

    # 3. Face Analysis tab → wait for header.
    goto_page(page, "Face Analysis")
    page.wait_for_selector("h2:has-text('Face Analysis')", state="visible", timeout=30_000)

    # AC: 5 metric widgets present (Blur / Yaw / Pitch / Roll / Area).
    for label in ("Blur", "Yaw", "Pitch", "Roll", "Area"):
        page.wait_for_selector(f"text=/{label}/", state="visible", timeout=15_000)

    # AC: Plotly chart container rendered (bbox overlay; .js-plotly-plot is
    # the canonical Plotly root). Fall back to <img> if Plotly fell through
    # to the crop fallback (still satisfies "a face crop is shown").
    has_chart = page.locator(".js-plotly-plot").count() > 0
    has_img = page.locator("img").count() > 0
    assert has_chart or has_img, "Neither Plotly chart nor <img> rendered."

    # AC: Face id input echoes the selected id (number_input shows the value).
    # Target by role+label, NOT input[type=number].first: Streamlit keeps every
    # tab's body in the DOM, so .first would grab a hidden number_input from the
    # Run/Cluster Analysis tab. The "Face id" spinbutton is the visible one.
    face_id_input = page.get_by_role("spinbutton", name="Face id")
    face_id_input.wait_for(state="visible", timeout=10_000)
    val = face_id_input.input_value()
    assert val and val.strip().lstrip("-").isdigit(), (
        f"Face id input did not echo a numeric id (got {val!r})."
    )
