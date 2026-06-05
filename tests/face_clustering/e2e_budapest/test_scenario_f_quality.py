"""Scenario F — Quality tab summary + per-gate chart over the reference run.

Loads the reference run via the History tab and opens the Quality tab.
Verifies the summary strip + per-gate stacked bar chart render with the
expected reject count band.

Click sequence (per spec-065 §"E2E contract"):
  1. Reference run loaded via the ``page_with_reference_run_loaded`` fixture
     (spec-067: query-param seed replaces the canvas dataframe row-pick).
  2. Quality tab

Assertions:
  1. Tab visible (h2 "Quality")
  2. Summary strip has >= 4 metric widgets
  3. Per-gate Plotly chart rendered (>= 1 trace / 1 bar)
  4. "Rejected" metric value within EXPECTED_REJECTED_BAND
     (220..240 = 340 total faces - 107 assigned ± 7)

What this catches:
  - QualityService → QualitySummary aggregation regressions
  - Repository.list_filter_decisions schema drift
  - render_quality_bar_chart short-circuit to st.info on empty input
  - Reference run's filter_decisions table getting wiped or dropped
"""
from __future__ import annotations

import re

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    EXPECTED_REJECTED_BAND,
)

pytestmark = pytest.mark.budapest


def test_scenario_f_quality_summary_and_chart(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    # 1. Reference run already seeded by the fixture (spec-067).
    # 2. Quality tab.
    page.get_by_role("tab", name="Quality").click()
    page.wait_for_selector("h2:has-text('Quality')", state="visible", timeout=30_000)

    # 3. >= 4 metric widgets (Items / Decisions / Rejected / Top reject gate / Pass rate)
    page.wait_for_selector("[data-testid='stMetric']", state="visible", timeout=30_000)
    n_metrics = page.locator("[data-testid='stMetric']").count()
    assert n_metrics >= 4, (
        f"Quality summary strip rendered {n_metrics} metrics; expected >= 4. "
        f"summary() likely returned empty or aggregation broke."
    )

    # 4. Plotly chart rendered with at least one bar.
    page.wait_for_selector(".js-plotly-plot", state="visible", timeout=15_000)
    n_traces = page.locator(".js-plotly-plot .trace").count()
    n_bars = page.locator(".js-plotly-plot .bars").count()
    assert n_traces >= 1 or n_bars >= 1, (
        f"Per-gate chart rendered no traces / bars (traces={n_traces}, bars={n_bars}). "
        f"render_quality_bar_chart likely fell through to st.info on empty input."
    )

    # 5. Rejected count in the expected band. Metrics: Items / Decisions /
    # Rejected / Top reject gate / Pass rate. "Rejected" is the 3rd metric;
    # parse its first integer.
    metric_texts = page.locator("[data-testid='stMetric']").all_text_contents()
    rejected_metric = next((t for t in metric_texts if "Rejected" in t and "rate" not in t.lower()), None)
    assert rejected_metric, (
        f"Could not find 'Rejected' metric in summary strip: {metric_texts!r}"
    )
    m = re.search(r"(\d+)", rejected_metric.replace("Rejected", "", 1))
    assert m, f"Could not parse rejected count from metric text: {rejected_metric!r}"
    rejected = int(m.group(1))
    lo, hi = EXPECTED_REJECTED_BAND
    assert lo <= rejected <= hi, (
        f"Quality 'Rejected' = {rejected}; expected in {EXPECTED_REJECTED_BAND}. "
        f"Either the filter pipeline drifted or the aggregation is wrong."
    )
