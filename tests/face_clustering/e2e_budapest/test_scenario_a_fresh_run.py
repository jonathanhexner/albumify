"""Scenario A — fresh pipeline run on Budapest + profile_4.

Click sequence:
  1. Open the "Run" tab.
  2. Fill the "Source" text input with `D:\\Budapest2025_Google`.
  3. Fill the "Album" text input with a unique e2e name.
  4. Open the profile selectbox; pick `profile_4`.
  5. Click the "Run pipeline" button.
  6. Wait up to 10 min for the success message ("Run complete — N clusters …").

Assertions (from baseline — must match the legacy reference run):
  - success message appears (pipeline completed)
  - n_clusters == 15
  - parsed n_clusters_from_message == EXPECTED_N_CLUSTERS

What this catches:
  - Pipeline regression (any commit that changes clustering output on this profile)
  - Run tab UI breakage (input fields, selectbox, button disabled-state)
  - Pipeline crash mid-run (no success message)
  - Profile loading bug (wrong K / threshold → different cluster count)

See ``README.md`` in this dir for the full functionality matrix.
"""
from __future__ import annotations

import re
import time

import pytest

from tests.face_clustering.e2e_budapest.conftest import (
    goto_page,
    EXPECTED_N_CLUSTERS, PIPELINE_TIMEOUT_S, PROFILES_V2_DIR, PROFILE_NAME, SOURCE_DIR,
)

pytestmark = pytest.mark.budapest


def test_scenario_a_fresh_run_produces_baseline_cluster_count(page):
    if not SOURCE_DIR.exists():
        pytest.skip(f"Source dir missing: {SOURCE_DIR}")
    if not (PROFILES_V2_DIR / PROFILE_NAME).exists():
        pytest.skip(f"Profile missing: {PROFILES_V2_DIR / PROFILE_NAME}")

    # 1. Run tab
    goto_page(page, "Run")

    # 2-3. Source + Album. Use exact labels — `get_by_label("Album")` collides
    # with the History tab's "Selected (all). Album" filter selectbox and the
    # search box's "album / run name / comment" hint (st.tabs is not lazy).
    page.get_by_label("Source image directory", exact=True).fill(str(SOURCE_DIR))
    page.get_by_label("Album name (required)", exact=True).fill(
        f"e2e_baseline_{int(time.time())}"
    )

    # 4. profile_4. The "Load profile" selectbox lives inside an
    # `st.expander("Profiles", expanded=False)` — open the expander first.
    # Use the accessible label, not `combobox.first` (ambiguous now that
    # History/Recluster tabs each render their own selectboxes; st.tabs
    # renders every body on every script run).
    page.get_by_role("button", name="Profiles").click()
    page.get_by_label("Load profile", exact=True).click()
    page.get_by_role("option", name="profile_4").first.click()
    page.get_by_role("button", name="Load", exact=True).click()

    # 5. Run pipeline
    page.get_by_role("button", name="Run pipeline").click()

    # 6. Wait for success
    page.wait_for_selector(
        "text=/Run complete.*clusters/i",
        state="visible",
        timeout=PIPELINE_TIMEOUT_S * 1000,
    )

    # Parse n_clusters out of the success message:
    #   "Run complete — N clusters from M faces across K images ..."
    msg = page.locator("text=/Run complete.*clusters/i").first.text_content() or ""
    m = re.search(r"(\d+)\s+clusters", msg)
    assert m, f"Could not parse n_clusters from success message: {msg!r}"
    n_clusters = int(m.group(1))
    assert n_clusters == EXPECTED_N_CLUSTERS, (
        f"Baseline regression: profile_4 against Budapest produced {n_clusters} "
        f"clusters; expected {EXPECTED_N_CLUSTERS}. See README.md §'Reference run'."
    )
