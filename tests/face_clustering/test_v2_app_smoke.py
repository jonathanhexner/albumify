"""End-to-end smoke for app/face_clustering_v2/main.py via Streamlit AppTest.

Exercises the EXACT page lifecycle the browser triggers:
  - All tab bodies execute on every script rerun (Streamlit's tab semantics).
  - st.session_state is honored (we seed v2_last_run_dir to point at the
    fixture run).
  - st.exception nodes count as failures (Streamlit's red traceback overlay).

This is the test class that would have caught SIGHTING-078 (the 2026-05-29
"RunStoreError: no clusters recorded for iteration 1" crash) before it
reached the user. The prior synthetic test exercised the Repository's class
methods in isolation; this one runs them through the actual Streamlit page
that bound the bug.

Opt-in via @pytest.mark.slow because AppTest cold-start is ~5s. Run with:
    .venv/Scripts/python -m pytest -m slow tests/face_clustering/test_v2_app_smoke.py -v

Skipped when no real run dir is available (CI without the user's machine).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from face_cluster.db.schema import SCHEMA_DDL, SCHEMA_VERSION

# Import the shared synthetic builder so this test is self-contained.
# Same fixture the Repository synthetic tests use, plus we apply the
# "merger ran but merged nothing" mutation so the run shape matches the
# user's failing real-world dirs (52a70e6f..., e51497605..., 6d59eb03...).
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
    _add_no_op_merge_round,
)


MAIN_PY = str(Path(__file__).resolve().parents[2] / "app" / "face_clustering_v2" / "main.py")


@pytest.fixture
def no_op_merge_run_dir(tmp_path: Path) -> Path:
    """Synthetic run dir with the exact shape that crashed SIGHTING-078:
    clusters @ iteration=0, merge_decisions @ iteration=1 with
    actually_merged=0 (merger considered + rejected every pair).
    """
    run_dir = _build_synthetic_run_dir(tmp_path)
    _add_no_op_merge_round(run_dir)
    return run_dir


@pytest.mark.slow
def test_main_page_renders_without_exception_when_no_run_loaded(tmp_path):
    """Baseline: with no run dir in session_state, the page must render
    without exception (the friendly "no completed run" info banner).
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN_PY)
    at.run(timeout=60)
    assert len(at.exception) == 0, (
        f"Page raised {len(at.exception)} exception(s) with no run loaded: "
        f"{[str(e.value) for e in at.exception]}"
    )


@pytest.mark.slow
def test_main_page_renders_without_exception_on_no_op_merge_run(no_op_merge_run_dir):
    """SIGHTING-078 regression: a completed run where the merger considered
    pairs but rejected all of them (very common on clean runs) used to
    crash the entire app because Cluster Analysis tab body fires on every
    script rerun regardless of which tab the user is looking at.

    Asserts that with this shape seeded into session_state, the page
    renders cleanly — no st.exception, no st.error on the Cluster Analysis
    tab path.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN_PY)
    at.session_state["v2_last_run_dir"] = str(no_op_merge_run_dir)
    at.run(timeout=60)

    exceptions = [str(e.value) for e in at.exception]
    errors = [str(e.value)[:200] for e in at.error]
    assert not exceptions, (
        f"Page raised {len(exceptions)} exception(s) on no-op-merge run dir:\n  "
        + "\n  ".join(exceptions)
    )
    # We accept st.info ("no run loaded" empty-state) but not st.error
    # (which would indicate the defensive _get_service try/except fired
    # — meaning a code path we thought was safe still crashed).
    assert not errors, (
        f"Page rendered st.error on no-op-merge run dir (defensive catch fired):\n  "
        + "\n  ".join(errors)
    )


@pytest.mark.slow
def test_main_page_renders_without_exception_on_empty_run_dir(tmp_path):
    """2026-05-29 earlier bug regression: spec-050 allocates a UUID run dir
    and writes v2_last_run_dir BEFORE the pipeline runs. If the user opens
    the app before the pipeline writes face_clustering.db, the resolver
    must skip the dir gracefully — no crash, friendly empty-state info
    banner instead.
    """
    from streamlit.testing.v1 import AppTest

    allocated = tmp_path / "deadbeef1234"
    allocated.mkdir()  # empty dir — no DB yet, simulates spec-050 pre-pipeline state

    at = AppTest.from_file(MAIN_PY)
    at.session_state["v2_last_run_dir"] = str(allocated)
    at.run(timeout=60)

    exceptions = [str(e.value) for e in at.exception]
    assert not exceptions, (
        f"Page raised {len(exceptions)} exception(s) on empty run dir:\n  "
        + "\n  ".join(exceptions)
    )
    # Expect at least one st.info ("no completed run available yet") on the page.
    assert len(at.info) >= 1, (
        "Empty run dir should produce a friendly st.info empty-state banner; "
        "got none."
    )
