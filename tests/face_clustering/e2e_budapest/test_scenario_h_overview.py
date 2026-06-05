"""Scenario H — Overview tab: run-level dashboard.

Overview reads the global action_log (not a single run), so no run needs
loading — but we reuse the reference-run fixture for a consistent app
state. The real action_log on the dev machine has many fc_app_v2 runs
(incl. the reference album), so the dashboard renders real bars.

Click sequence (spec-066 §"E2E contract"):
  1. App open (reference run seeded by the fixture; harmless for Overview).
  2. Overview tab -> wait for header.

Assertions:
  1. Overview tab visible (h2 "Overview").
  2. The 4-metric strip rendered (Total runs / Total faces ever /
     Avg n_clusters / Last run labels all present).
  3. The "Runs per album" chart rendered (a Plotly plot is present).
  4. The "Runs per status" chart rendered (the spec-066 decision-A swap of
     the original per-profile chart, which had no data).

What this catches:
  - OverviewService.compute_dashboard crashing on the real action_log shape.
  - dashboard_charts failing to render (Plotly key collisions, empty data).
  - the 4-metric strip silently not painting.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.budapest


def test_scenario_h_overview(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    # 2. Overview tab -> header. 60s: a tab-click re-runs all 9 tab bodies.
    page.get_by_role("tab", name="Overview").click()
    page.wait_for_selector("h2:has-text('Overview')", state="visible", timeout=60_000)

    # AC2: the 4-metric strip.
    for label in ("Total runs", "Total faces ever", "Avg n_clusters", "Last run"):
        page.wait_for_selector(f"text=/{label}/", state="visible", timeout=30_000)

    # AC3 + AC4: both bar charts rendered.
    page.wait_for_selector("text=/Runs per album/", state="visible", timeout=30_000)
    page.wait_for_selector("text=/Runs per status/", state="visible", timeout=30_000)
    assert page.locator(".js-plotly-plot").count() >= 1, "No Plotly chart rendered on Overview."
