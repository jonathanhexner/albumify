"""spec-079 — single source of truth for the Budapest equivalence anchor.

Both spec-079 test suites import these constants:

* ``tests/api/`` (backend / endpoint tests, FastAPI TestClient)
* ``tests/streamlit/e2e_albumify/`` (slim Playwright)

The numbers are the verified Face-Clustering-v2 reference run on the same
album + profile. Albumify, sharing the pipeline, is held to them after every
refactor stage. If a stage changes these, the stage is wrong — not the anchor.

Source of the numbers: reference run ``a588521b993540e78f40935ecdb21b58``
(producer ``fc_app_v2``, profile ``profile_5.json``), re-verified 2026-06-06
by ``scripts/run_profile.py`` which reproduced the exact cluster shape from the
profile alone (Step 0 PASS). The run dir's own ``config_json`` is empty — the
profile name was recovered from the central ``action_log``.
"""
from __future__ import annotations

from pathlib import Path

# --- Inputs --------------------------------------------------------------
SOURCE_DIR = Path(r"D:\Budapest2025_Google")
# FC v2 clustering profile (flat FCParams JSON). Translated to API step
# configs via ``FCParams.load(PROFILE_PATH).to_step_configs()`` so Albumify
# runs the SAME clustering knobs FC v2 used.
PROFILE_PATH = Path.home() / ".sim_bench" / "profiles_v2" / "profile_5.json"

# FC v2 reference run that produced the gold cluster shape.
REFERENCE_RUN_ID = "a588521b993540e78f40935ecdb21b58"
REFERENCE_RUN_DIR = Path.home() / ".sim_bench" / "runs" / REFERENCE_RUN_ID

# --- Gold numbers (FC v2 reference, profile_5) ---------------------------
EXPECTED_N_CLUSTERS = 8
EXPECTED_N_FACES_TOTAL = 340
EXPECTED_N_FACES_ASSIGNED = 72   # sum of EXPECTED_CLUSTER_SIZES
EXPECTED_CLUSTER_SIZES = [26, 20, 12, 7, 3, 2, 2]

# Documented tolerance bands so minor gate-count drift across pipeline
# iterations never red-flags a good run.
EXPECTED_REJECTED_BAND = (255, 285)   # total - assigned ~= 268, ±
EXPECTED_CLUSTER_COUNT_BAND = (6, 10)


def assert_matches_anchor(n_clusters: int, n_assigned: int) -> None:
    """Raise AssertionError with a diagnostic if counts miss the anchor band."""
    lo, hi = EXPECTED_CLUSTER_COUNT_BAND
    assert lo <= n_clusters <= hi, (
        f"n_clusters={n_clusters} outside FC v2 band {EXPECTED_CLUSTER_COUNT_BAND} "
        f"(reference={EXPECTED_N_CLUSTERS}). Albumify diverged from the shared pipeline."
    )
    rlo, rhi = EXPECTED_REJECTED_BAND
    rejected = EXPECTED_N_FACES_TOTAL - n_assigned
    assert rlo <= rejected <= rhi, (
        f"rejected={rejected} outside band {EXPECTED_REJECTED_BAND} "
        f"(assigned={n_assigned}, total={EXPECTED_N_FACES_TOTAL})."
    )
