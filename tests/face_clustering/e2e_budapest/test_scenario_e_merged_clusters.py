"""Scenario E — Merged Clusters tab viewer over the reference run.

Loads the reference run via the query-param seed and opens the Merged
Clusters tab. The reference run's merger ran at least one iteration, so
``merge_decisions`` is populated (telemetry confirms n_rows=3).

Click sequence (per spec-065 §"E2E contract"):
  1. Reference run loaded via the ``page_with_reference_run_loaded`` fixture
     (spec-067: query-param seed replaces the canvas dataframe row-pick).
  2. Merged Clusters tab

Assertion (browser layer, spec-067 §"Test coverage strategy per tab"):
  1. Tab visible (h2 "Merged Clusters")
  2. The merge_decisions table *container* renders. ``merged_clusters_tab``
     returns ``st.info`` BEFORE ``render_run_table`` when the merge log is
     empty, so a present-and-visible ``stDataFrame`` container proves >= 1
     row reached the UI.

Why not assert on rows/cells/detail-panel here: ``render_run_table`` is a
canvas ``st.dataframe`` (glide-data-grid) — the cells are painted on
``<canvas>`` and are NOT Playwright-addressable (the same constraint as the
History picker, SIGHTING-091). Row count + field correctness
(cluster_a / cluster_b / actually_merged / exemplar_dist / support) are
covered at the service layer by
``tests/face_clustering/views/test_merged_clusters_service_synthetic.py``,
including ``test_real_fixture_list_merge_decisions`` against the budapest run.

What this catches:
  - Repository -> Service -> Tab wiring producing an empty table
    (st.info fallback) when the merge log is non-empty.
  - The tab failing to render at all.
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import goto_page

pytestmark = pytest.mark.budapest


def test_scenario_e_merged_clusters_viewer(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    # 1. Reference run already seeded by the fixture (spec-067).
    # 2. Merged Clusters tab.
    goto_page(page, "Merged Clusters")
    page.wait_for_selector("h2:has-text('Merged Clusters')", state="visible", timeout=30_000)

    # 3. Table container present in the active tab. ``:visible`` scopes past the
    # other tabs' stDataFrame containers (History etc.) which are in the DOM but
    # hidden — Streamlit renders every tab body each run.
    page.wait_for_selector(
        "[data-testid='stDataFrame']:visible", state="visible", timeout=15_000
    )
