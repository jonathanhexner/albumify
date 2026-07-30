"""Scenario J — Face Metrics grid: clicking a face opens Face Analysis.

spec-083: the Face Metrics drill-in was a canvas ``st.dataframe`` row-select
whose only working target was a ~20px checkbox column — users reported "I can't
click on faces" repeatedly. The Grid layout (default) renders a real
``st.button("Open")`` per face (Playwright-addressable), and clicking it
``navigate_to("Face Analysis")``. This scenario guards that path so the
clickability can never silently regress again.

Click sequence:
  1. Reference run seeded by the fixture.
  2. Face Metrics tab -> wait for a grid Open button -> click it.
  3. The click itself navigates to Face Analysis (no manual tab switch).

Assertions:
  1. Face Analysis header visible after the click (proves auto-navigation).
  2. Face id spinbutton echoes a numeric id (proves selected_face_id wired).
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import goto_page

pytestmark = pytest.mark.budapest


def test_scenario_j_face_metrics_click(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    goto_page(page, "Face Metrics")
    page.wait_for_selector("h2:has-text('Face Metrics')", state="visible")
    open_btn = page.get_by_role("button", name="Open").first
    open_btn.wait_for(state="visible", timeout=60_000)
    open_btn.click()

    # Clicking Open calls navigate_to("Face Analysis") -> the page switches on
    # its own. If the wiring breaks we stay on Face Metrics and this times out.
    page.wait_for_selector("h2:has-text('Face Analysis')", state="visible", timeout=30_000)

    face_id_input = page.get_by_role("spinbutton", name="Face id")
    face_id_input.wait_for(state="visible", timeout=10_000)
    val = face_id_input.input_value()
    assert val and val.strip().lstrip("-").isdigit(), (
        f"Face id input did not echo a numeric id (got {val!r})."
    )
