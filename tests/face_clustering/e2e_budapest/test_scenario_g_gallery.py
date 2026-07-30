"""Scenario G — Gallery tab: browse clusters as thumbnail strips.

Loads the reference run, opens the Gallery tab, and verifies the strips
render with real face crops + the cross-tab "Open in Cluster Analysis"
nav works.

Click sequence (spec-066 §"E2E contract", D4 — the test clicks the tab;
Streamlit has no programmatic tab-switch):
  1. Reference run seeded by the ``page_with_reference_run_loaded`` fixture.
  2. Gallery tab -> wait for header -> assert cluster rows + thumbnails.
  3. Click cluster 1's "Open in Cluster Analysis" -> Cluster Analysis tab
     -> assert the picker selected cluster 1.

Assertions:
  1. Gallery tab visible (h2 "Gallery").
  2. >= 1 cluster row with >= 1 face thumbnail (<img>).
  3. The largest cluster ("Cluster 1") row is present (size 35 > 8 -> 8 thumbs).
  4. "Open in Cluster Analysis" selects cluster 1: after clicking it and
     opening Cluster Analysis, the picker shows "Cluster 1".

What this catches:
  - exemplar_face_ids / crop_path returning bad paths (no <img> renders).
  - cluster_strip's Open button not writing selected_cluster.
  - SIGHTING-079 class regression on the Gallery tab.
"""
from __future__ import annotations

import pytest

from tests.face_clustering.e2e_budapest.conftest import goto_page

pytestmark = pytest.mark.budapest


def test_scenario_g_gallery(page_with_reference_run_loaded):
    page = page_with_reference_run_loaded

    # 2. Gallery tab -> header. 60s: Streamlit re-renders all tab bodies per
    # rerun. (Blocked while the Face Metrics tab embeds 340 inline base64
    # crops before Gallery in the tab order — see spec-066 REVIEW / SIGHTING.)
    goto_page(page, "Gallery")
    page.wait_for_selector("h2:has-text('Gallery')", state="visible", timeout=60_000)

    # AC1/AC3: the largest cluster row is present.
    page.wait_for_selector("text=/Cluster 1\\b/", state="visible", timeout=60_000)

    # AC2: at least one VISIBLE thumbnail. Scope to ``:visible`` — Cluster
    # Analysis / Face Metrics also render <img> crops, earlier in DOM order,
    # but in hidden tab panels; an unscoped "img" wait would lock onto one of
    # those hidden images and never go visible.
    page.locator("css=img:visible").first.wait_for(state="visible", timeout=30_000)
    assert page.locator("css=img:visible").count() >= 1, "No visible Gallery thumbnails."

    # AC4: cross-tab nav. The Gallery sorts size-desc, so the first
    # "Open in Cluster Analysis" button belongs to cluster 1.
    open_btn = page.get_by_role("button", name="Open in Cluster Analysis").first
    open_btn.wait_for(state="visible", timeout=30_000)
    open_btn.click()

    goto_page(page, "Cluster Analysis")
    page.wait_for_selector("h2:has-text('Cluster Analysis')", state="visible", timeout=60_000)
    # The picker's selectbox should now read "Cluster 1 (...)".
    page.wait_for_selector("text=/Cluster 1 \\(/", state="visible", timeout=60_000)
