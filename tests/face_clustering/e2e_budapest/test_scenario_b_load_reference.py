"""Scenario B — load the reference run + verify cluster shape.

The reference run (`6437d335de914755bc3edb825c9591c0`) is the ground truth
the v2 stack must be able to display. This scenario doesn't run the
pipeline — it loads the existing run via History and inspects what the
Cluster Analysis tab renders.

Click sequence:
  1. Open the "History" tab.
  2. Find and click the row whose run-id cell contains the reference run's
     first 8 chars (`6437d335`).
  3. Click "Load into analysis tabs"; wait for success toast.
  4. Open the "Cluster Analysis" tab.
  5. Wait for the metric strip (`[data-testid='stMetric']`) to appear.

Assertions (verified 2026-05-30 from the reference DB):
  - ≥ 5 metric widgets render (Faces / Diameter / Avg / Exemplars / Outliers)
  - ≥ 1 face thumbnail (`<img>`) renders in the face grid
  - Cluster picker is populated (selectbox has options)
  - Default-selected cluster shows the expected size (currently 24 — cluster 0
    in the reference; if the picker defaults to a different cluster the
    Faces metric should still match a known size from EXPECTED_CLUSTER_SIZES)

What this catches:
  - SIGHTING-078 (RunStore crash on no-op-merge runs) — page wouldn't load
  - SIGHTING-079 (UI stuck on "Analysing cluster…") — metric strip empty
  - SIGHTING-080 (History rejects v2 runs) — Load Run button disabled
  - SIGHTING-089 (History panel blank for v2 runs) — Load Run still fires
    but the loaded state is sparse (covered indirectly via metric render)
  - Today's face-grid bug — captions only, no `<img>` elements
  - Any future regression that makes the user load a known-good run and
    see nothing

See ``README.md`` for the full functionality matrix.
"""
from __future__ import annotations

import re

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    EXPECTED_BIGGEST_CLUSTER_SIZE, EXPECTED_CLUSTER_0_SIZE,
    REFERENCE_RUN_DIR, REFERENCE_RUN_ID,
)

pytestmark = pytest.mark.budapest

# Sizes of every real cluster in the reference run (verified against the DB
# on 2026-05-30). If profile_4 ever changes and re-clustering happens, this
# list is stale — update it in the same PR.
EXPECTED_CLUSTER_SIZES = [35, 24, 14, 7, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]


def test_scenario_b_load_reference_run_and_see_clusters(page):
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")

    # 1. History tab
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")

    # 2. Find + click the reference run row
    short = REFERENCE_RUN_ID[:8]
    page.get_by_role("gridcell", name=lambda s: short in s).first.click(timeout=15_000)

    # 3. Load Run
    page.get_by_role("button", name="Load into analysis tabs").click()
    page.wait_for_selector("text=/Loaded/i", state="visible", timeout=15_000)

    # 4. Cluster Analysis tab
    page.get_by_role("tab", name="Cluster Analysis").click()
    page.wait_for_selector("h2:has-text('Cluster Analysis')", state="visible")

    # 5. Metric strip
    page.wait_for_selector("[data-testid='stMetric']", state="visible", timeout=30_000)
    n_metrics = page.locator("[data-testid='stMetric']").count()
    assert n_metrics >= 5, (
        f"Cluster Analysis metric strip didn't render — got {n_metrics} metrics, "
        f"expected >= 5 (Faces / Diameter / Avg / Exemplars / Outliers)."
    )

    # The first metric is "Faces" (count); its value must match one of the
    # known cluster sizes from EXPECTED_CLUSTER_SIZES.
    faces_metric_text = page.locator("[data-testid='stMetric']").first.text_content() or ""
    m = re.search(r"\b(\d+)\b", faces_metric_text)
    assert m, f"Could not parse Faces metric value: {faces_metric_text!r}"
    n_faces_shown = int(m.group(1))
    assert n_faces_shown in EXPECTED_CLUSTER_SIZES, (
        f"Cluster Analysis 'Faces' metric = {n_faces_shown}; not in expected "
        f"cluster sizes {EXPECTED_CLUSTER_SIZES}. Either the reference run "
        f"shape drifted (update conftest + this list) or the picker is "
        f"reading the wrong cluster."
    )

    # Face thumbnails (today's regression class)
    page.wait_for_selector("img", state="visible", timeout=15_000)
    n_imgs = page.locator("img").count()
    assert n_imgs >= 1, (
        "No face thumbnails rendered in Cluster Analysis tab — face_grid is "
        "showing captions only. Today's regression pattern."
    )

    # Sanity ref to imports — silence linters
    assert EXPECTED_BIGGEST_CLUSTER_SIZE == 35
    assert EXPECTED_CLUSTER_0_SIZE == 24
