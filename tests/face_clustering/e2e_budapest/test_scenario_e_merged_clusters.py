"""Scenario E — Merged Clusters tab viewer over the reference run.

Loads the reference run via the History tab and opens the Merged
Clusters tab. The reference run's merger ran at least one iteration,
so ``merge_decisions`` is populated.

Click sequence (per spec-065 §"E2E contract"):
  1. History tab → click row containing `6437d335` → "Load into analysis tabs"
  2. Merged Clusters tab

Assertions:
  1. Tab visible (h2 "Merged Clusters")
  2. >= EXPECTED_MERGE_DECISIONS_MIN_ROWS rows in the table
  3. Clicking the first row reveals a detail panel (st.json block)
  4. Detail panel exposes the canonical fields:
     cluster_a / cluster_b / actually_merged / exemplar_dist / support

What this catches:
  - Repository → Service → Tab wiring for merge_decisions
  - Tab forgetting to render the detail panel on row select
  - Schema drift on MergeDecisionRow that breaks asdict
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    EXPECTED_MERGE_DECISIONS_MIN_ROWS,
    REFERENCE_RUN_DIR,
    REFERENCE_RUN_ID,
)

pytestmark = pytest.mark.budapest

# Fields that must surface in the detail panel JSON.
_REQUIRED_DETAIL_FIELDS = (
    "cluster_a", "cluster_b", "actually_merged", "exemplar_dist", "support",
)


def test_scenario_e_merged_clusters_viewer(page):
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")

    # 1. History → load reference run.
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")
    short = REFERENCE_RUN_ID[:8]
    page.get_by_role("gridcell", name=lambda s: short in s).first.click(timeout=15_000)
    page.get_by_role("button", name="Load into analysis tabs").click()
    page.wait_for_selector("text=/Loaded/i", state="visible", timeout=15_000)

    # 2. Merged Clusters tab.
    page.get_by_role("tab", name="Merged Clusters").click()
    page.wait_for_selector("h2:has-text('Merged Clusters')", state="visible", timeout=30_000)

    # 3. Show "all" rows (default selectbox value) and verify the table populates.
    # render_run_table places its dataframe behind data-testid=stDataFrame; if it
    # short-circuited with st.info we'd see no gridcells.
    page.wait_for_selector("[role='gridcell']", state="visible", timeout=15_000)
    n_cells = page.locator("[role='gridcell']").count()
    assert n_cells >= EXPECTED_MERGE_DECISIONS_MIN_ROWS, (
        f"Merged Clusters table rendered {n_cells} cells; expected "
        f">= {EXPECTED_MERGE_DECISIONS_MIN_ROWS}. Either the merge log is "
        f"unexpectedly empty or render_run_table fell through to st.info."
    )

    # 4. Click first gridcell → detail panel opens (st.json block + subheader).
    page.locator("[role='gridcell']").first.click()
    page.wait_for_selector("text=/Pair \\(cluster_a=/", state="visible", timeout=10_000)

    # 5. st.json renders the asdict() payload as a JSON tree; assert each
    # canonical field label is present in the panel content.
    body_text = page.locator("body").text_content() or ""
    missing = [f for f in _REQUIRED_DETAIL_FIELDS if f not in body_text]
    assert not missing, (
        f"Detail panel missing expected fields: {missing}. "
        f"Body snippet: {body_text[:500]!r}"
    )
