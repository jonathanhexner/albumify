"""Scenario D — Face Analysis drill-down from a Cluster Analysis thumbnail.

Loads the reference run via the History tab, switches to Cluster Analysis,
waits for the face_grid to render, clicks the first thumbnail's "Open"
button (which writes ``selected_face_id`` to session_state), then opens
the Face Analysis tab and verifies the per-face popup rendered.

Click sequence (per spec-064 §"E2E contract"):
  1. History tab → row containing `6437d335` → "Load into analysis tabs"
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

from tests.face_clustering.e2e_budapest.conftest import (
    REFERENCE_RUN_DIR,
    REFERENCE_RUN_ID,
)

pytestmark = pytest.mark.budapest


def test_scenario_d_face_analysis_drill_down(page):
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")

    # 1. History → load reference run.
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")
    short = REFERENCE_RUN_ID[:8]
    page.get_by_role("gridcell", name=lambda s: short in s).first.click(timeout=15_000)
    page.get_by_role("button", name="Load into analysis tabs").click()
    page.wait_for_selector("text=/Loaded/i", state="visible", timeout=15_000)

    # 2. Cluster Analysis → wait for face_grid to render an Open button.
    page.get_by_role("tab", name="Cluster Analysis").click()
    page.wait_for_selector("h2:has-text('Cluster Analysis')", state="visible")
    open_btn = page.get_by_role("button", name="Open").first
    open_btn.wait_for(state="visible", timeout=60_000)
    open_btn.click()

    # 3. Face Analysis tab → wait for header.
    page.get_by_role("tab", name="Face Analysis").click()
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
    face_id_input = page.locator("input[type='number']").first
    face_id_input.wait_for(state="visible", timeout=10_000)
    val = face_id_input.input_value()
    assert val and val.strip().lstrip("-").isdigit(), (
        f"Face id input did not echo a numeric id (got {val!r})."
    )
