"""spec-067 / SIGHTING-091 — query-param seeder on app/face_clustering_v2/main.py.

The v2 History tab's run-picker is a canvas-rendered ``st.dataframe`` that
Playwright cannot click. Rather than change that UI, ``main.py`` honors a
small allowlist of query params on first render so the budapest e2e suite
(and real users via deep-link) can seed ``current_run_dir`` directly. These
tests pin that contract:

  - AC1: ``?current_run_dir=X`` lands in ``session_state['current_run_dir']``.
  - AC6: with no query params the seeder writes nothing (opt-in, no
    behaviour change for the default app).

AppTest cold-start is ~5s, so these are ``@pytest.mark.slow`` like the rest
of the v2 AppTest smokes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

MAIN_PY = str(
    Path(__file__).resolve().parents[2] / "app" / "face_clustering_v2" / "main.py"
)


@pytest.mark.slow
def test_current_run_dir_query_param_seeds_session_state():
    """AC1: ``?current_run_dir=<path>`` is mirrored into session_state
    (both ``current_run_dir`` and ``active_run_dir``, matching the Load
    button) and the page renders without exception."""
    from streamlit.testing.v1 import AppTest

    target = r"C:\Users\example\.sim_bench\runs\deadbeef"
    at = AppTest.from_file(MAIN_PY)
    at.query_params["current_run_dir"] = target
    at.run(timeout=60)

    assert len(at.exception) == 0, (
        f"Seeding raised: {[str(e.value) for e in at.exception]}"
    )
    assert at.session_state["current_run_dir"] == target
    assert at.session_state["active_run_dir"] == target


@pytest.mark.slow
def test_no_query_params_leaves_run_dir_unset():
    """AC6: opt-in. With no query params the seeder must not write
    ``current_run_dir`` — the app behaves exactly as before spec-067."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN_PY)
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert "current_run_dir" not in at.session_state
    assert "active_run_dir" not in at.session_state
    # Sentinel is set so re-runs are idempotent.
    assert at.session_state["_qp_seeded"] is True


@pytest.mark.slow
def test_selected_face_id_query_param_is_int_parsed():
    """Allowlisted ``?selected_face_id=<int>`` seeds the Face Analysis
    default; a non-int value is ignored rather than crashing."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN_PY)
    at.query_params["selected_face_id"] = "42"
    at.run(timeout=60)
    assert len(at.exception) == 0
    assert at.session_state["selected_face_id"] == 42

    at_bad = AppTest.from_file(MAIN_PY)
    at_bad.query_params["selected_face_id"] = "not-an-int"
    at_bad.run(timeout=60)
    assert len(at_bad.exception) == 0
    assert "selected_face_id" not in at_bad.session_state
