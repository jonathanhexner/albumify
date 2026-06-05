"""Scenario I — Merged Clusters detail panel (gate badges + pair crops).
(G/H reserved for Gallery/Overview, spec-066.)

spec-071: V2's Merged Clusters now shows, per merge pair, the 5 gate verdicts
and the two clusters' faces side by side. The detail panel renders on a
canvas-`st.dataframe` row-select, which Playwright can't click (SIGHTING-091),
so we drive it the spec-067 way: seed ``?selected_merge_pair=a,b`` (a real pair
read from the reference run) and assert the panel paints.

Click sequence:
  1. Page seeded with current_run_dir + selected_merge_pair (real pair).
  2. Merged Clusters tab.

Assertions:
  1. Detail panel present (subheader "Pair (cluster_a=…").
  2. Gate-badge names render (cross … diameter) — proves render_merge_gate_badges.
  3. >= 1 face crop <img> renders — proves render_cluster_pair_crops resolved faces.

What this catches:
  - Service gate_badges / pair_faces wiring to the components.
  - The components failing to paint badges or crops.
"""
from __future__ import annotations

import urllib.parse

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    APP_URL, PAGE_TIMEOUT_MS, REFERENCE_RUN_DIR,
)

pytestmark = pytest.mark.budapest


def _first_merge_pair():
    """Read a real (cluster_a, cluster_b) from the reference run's merge log."""
    from face_cluster.views.merged_clusters import MergedClustersService
    from sim_bench.db.face_clustering.cluster_analysis_repo import (
        ClusterAnalysisRepoConfig, ClusterAnalysisRepository,
    )
    svc = MergedClustersService(
        ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=REFERENCE_RUN_DIR))
    )
    rows = svc.list_merge_decisions()
    return rows[0] if rows else None


def test_scenario_i_merged_clusters_detail(page):
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")
    row = _first_merge_pair()
    if row is None:
        pytest.skip("Reference run has no merge_decisions.")

    # 1. Seed run + pair (bypasses the un-clickable canvas row-pick).
    seeded = (
        f"{APP_URL}?current_run_dir={urllib.parse.quote(str(REFERENCE_RUN_DIR))}"
        f"&selected_merge_pair={row.cluster_a},{row.cluster_b}"
    )
    page.goto(seeded)
    page.wait_for_selector("h1", state="visible", timeout=PAGE_TIMEOUT_MS)

    # 2. Merged Clusters tab.
    page.get_by_role("tab", name="Merged Clusters").click()
    page.wait_for_selector("h2:has-text('Merged Clusters')", state="visible", timeout=30_000)

    # AC1: the seeded detail panel painted (real-DOM subheader).
    page.wait_for_selector(r"text=/Pair \(cluster_a=/", state="visible", timeout=15_000)

    # Use VISIBLE body text (inner_text excludes hidden tabs + the canvas table's
    # off-screen a11y cells), so these assertions read only the active detail
    # panel — avoiding the SIGHTING-091 canvas-visibility trap.
    body = page.locator("body").inner_text()

    # AC2: gate badges + numbers caption (spec-071 detail panel). "diameter" is
    # NOT a merge-table column, so its presence proves the badge/caption, not
    # the table.
    assert "exemplar_dist=" in body, f"detail numbers caption missing; body[:300]={body[:300]!r}"
    for gate in ("cross", "exemplar", "support", "margin", "diameter"):
        assert gate in body, f"gate badge '{gate}' missing from detail panel"

    # AC3: pair-crop captions + at least one visible crop image.
    assert "cluster_a =" in body and "cluster_b =" in body, "pair-crop captions missing"
    assert page.locator("img:visible").count() >= 1, "No visible pair-crop images."
