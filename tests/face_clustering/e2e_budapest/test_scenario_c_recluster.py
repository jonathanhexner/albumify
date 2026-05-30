"""Scenario C — recluster the reference run from the Recluster tab.

Loads the reference run (`6437d335de914755bc3edb825c9591c0`) via the
History tab, switches to the Recluster tab, leaves the default params
in place, and clicks "Run recluster". Verifies a new snapshot run dir
is written with the right parent_run_id and a cluster count within
the expected band.

Click sequence (per spec-063 §"E2E contract"):
  1. History tab → click row containing `6437d335` → "Load into analysis tabs"
  2. Recluster tab → leave default params → "Run recluster" → wait for spinner

Assertions (per spec-063 §"E2E contract"):
  1. "Recluster complete" message visible
  2. A new run dir exists under `~/.sim_bench/runs/` whose pipeline_run.json
     has parent_run_id == 6437d335de914755bc3edb825c9591c0
  3. Parsed n_clusters in EXPECTED_RECLUSTER_BAND (12..18)
  4. The new run is visible in History tab on rerun

What this catches:
  - Recluster wiring corrupting the input face_records
  - Producer chain being invoked accidentally (would take ~10 min instead of ~15 s)
  - parent_run_id lineage broken (History tab loses the parent linkage)
  - SIGHTING-079 class regression on the Recluster tab (tab stuck on spinner)

See ``README.md`` for the full functionality matrix.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    EXPECTED_RECLUSTER_BAND,
    REFERENCE_RUN_DIR,
    REFERENCE_RUN_ID,
)

pytestmark = pytest.mark.budapest


def _runs_base() -> Path:
    return Path.home() / ".sim_bench" / "runs"


def test_scenario_c_recluster_reference_run(page):
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(f"Reference run missing: {REFERENCE_RUN_DIR}")

    # Snapshot existing run dirs so we can identify the new one afterwards.
    runs_base = _runs_base()
    pre_existing = {p.name for p in runs_base.iterdir() if p.is_dir()} if runs_base.exists() else set()

    # 1. History tab → load reference run.
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")

    short = REFERENCE_RUN_ID[:8]
    page.get_by_role("gridcell", name=short).first.click(timeout=15_000)
    page.get_by_role("button", name="Load into analysis tabs").click()
    page.wait_for_selector("text=/Loaded/i", state="visible", timeout=15_000)

    # 2. Recluster tab → leave defaults → Run recluster.
    page.get_by_role("tab", name="Recluster").click()
    page.wait_for_selector("h2:has-text('Recluster')", state="visible")

    # Wait for the picker to populate, then click Run.
    page.wait_for_selector("text=/Prior run/i", state="visible", timeout=10_000)
    page.get_by_role("button", name="Run recluster").click()

    # Recluster on the Budapest reference takes ~5-15 s on the test box.
    # Be generous: up to 5 min wall-clock so a slow CI doesn't false-fail.
    page.wait_for_selector(
        "text=/Recluster complete/i", state="visible", timeout=300_000,
    )

    # 3. Find the new run dir on disk.
    post_existing = {p.name for p in runs_base.iterdir() if p.is_dir()}
    new_dirs = post_existing - pre_existing
    assert new_dirs, (
        f"No new run dir under {runs_base} after recluster. "
        f"Existing: {sorted(pre_existing)}"
    )
    # If multiple were created (shouldn't happen), pick the most recent.
    new_dir = max(
        (runs_base / name for name in new_dirs),
        key=lambda p: p.stat().st_mtime,
    )

    # 4. Verify pipeline_run.json has the expected parent_run_id.
    pr_path = new_dir / "pipeline_run.json"
    assert pr_path.is_file(), f"pipeline_run.json missing in {new_dir}"
    payload = json.loads(pr_path.read_text(encoding="utf-8"))
    assert payload.get("parent_run_id") == REFERENCE_RUN_ID, (
        f"parent_run_id in {pr_path} = {payload.get('parent_run_id')!r}; "
        f"expected {REFERENCE_RUN_ID!r}"
    )

    # 5. Parse n_clusters from the success message and check the band.
    success_text = page.locator("text=/Recluster complete/i").first.text_content() or ""
    m = re.search(r"(\d+)\s+clusters", success_text)
    assert m, f"Could not parse n_clusters from success message: {success_text!r}"
    n_clusters = int(m.group(1))
    lo, hi = EXPECTED_RECLUSTER_BAND
    assert lo <= n_clusters <= hi, (
        f"Reclustered n_clusters = {n_clusters}; expected in {EXPECTED_RECLUSTER_BAND}. "
        f"Either the algorithm drifted or the recluster wiring corrupted the input."
    )

    # 6. Switch to History tab and verify the new run is listed.
    # Streamlit needs a beat to refresh the table after a writeback.
    time.sleep(2)
    page.get_by_role("tab", name="History").click()
    page.wait_for_selector("h2:has-text('History')", state="visible")
    short_new = new_dir.name[:8]
    page.wait_for_selector(f"text=/{short_new}/", state="visible", timeout=15_000)
