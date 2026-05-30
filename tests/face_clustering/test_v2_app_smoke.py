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

from sim_bench.run_db._schema import SCHEMA_DDL, SCHEMA_VERSION

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
def test_history_tab_recognizes_v2_run_as_loadable(no_op_merge_run_dir, isolate_action_log_db):
    """SIGHTING-080 regression: the History tab's "Load Run" button must be
    enabled for a v2 run (face_clustering.db at top level, no legacy CSVs).

    Before the 2026-05-29 fix, `_REQUIRED_ARTIFACTS` was a hardcoded legacy
    CSV trio. Every v2 run failed the artifact gate — Load Run button stayed
    disabled with "missing required artifacts (faces.csv, …)" warning.

    Test does the load end-to-end through HistoryService (the same code
    path the load_button calls when clicked). Asserts:
    - has_required_artifacts is True for a v5 run dir
    - HistoryService.load_run returns a typed LoadedRun without raising
    """
    from face_cluster.run_history_db import init_table
    from face_cluster.views.history import HistoryService
    from tests.face_clustering.views._seed import insert_action

    init_table(db_path=isolate_action_log_db)
    rid = insert_action(
        isolate_action_log_db,
        status="complete",
        output_dir=str(no_op_merge_run_dir),
        source_album="SIGHTING-080-regression",
        producer="fc_app_v2",
        action_type="fc_app_v2_run",
    )

    service = HistoryService()
    detail = service.get_run_detail(rid)
    assert detail.has_required_artifacts is True, (
        f"History tab thinks v2 run is missing artifacts. "
        f"run_dir={no_op_merge_run_dir}, contents={[p.name for p in no_op_merge_run_dir.iterdir()]}"
    )

    loaded = service.load_run(rid)
    assert loaded.output_dir == no_op_merge_run_dir
    assert loaded.pipeline_result is not None


@pytest.mark.slow
def test_history_tab_renders_without_exception_with_v2_run_seeded(
    no_op_merge_run_dir, isolate_action_log_db
):
    """SIGHTING-080 regression at the page level: with a v2 run in
    action_log, the History tab renders without raising. Previously, even
    rendering the runs table was fine — the bug only fired when the user
    clicked into a row and the load_button's gate evaluated has_required_artifacts.

    AppTest can't programmatically simulate a `st.dataframe` row selection
    (that requires browser interaction), so we assert the necessary
    invariants: (1) page renders cleanly, (2) no warning containing the
    legacy CSV trio appears anywhere (which would mean the load_button
    fired its old broken warning text on some auto-rendered path).
    """
    from face_cluster.run_history_db import init_table
    from streamlit.testing.v1 import AppTest
    from tests.face_clustering.views._seed import insert_action

    init_table(db_path=isolate_action_log_db)
    insert_action(
        isolate_action_log_db,
        status="complete",
        output_dir=str(no_op_merge_run_dir),
        source_album="SIGHTING-080-page-render",
        producer="fc_app_v2",
        action_type="fc_app_v2_run",
    )

    at = AppTest.from_file(MAIN_PY)
    at.run(timeout=60)

    exceptions = [str(e.value) for e in at.exception]
    assert not exceptions, (
        f"Page raised {len(exceptions)} exception(s) with v2 run seeded:\n  "
        + "\n  ".join(exceptions)
    )

    legacy_warnings = [
        w.value for w in at.warning
        if "faces.csv" in w.value or "clusters.csv" in w.value
    ]
    assert not legacy_warnings, (
        "load_button warning still references the legacy CSV trio "
        "(SIGHTING-080 regressed): " + " | ".join(legacy_warnings)
    )


@pytest.mark.slow
def test_cluster_analysis_face_grid_resolves_real_crop_thumbnails(no_op_merge_run_dir):
    """STOPGAP regression: face_grid renders thumbnails when the writer's
    filename pattern (``face_{id:04d}_aligned.jpg``) exists on disk under
    ``crops/``. The proper data flow (FaceRow.crop_path surfaced from
    FaceRecord) is deferred until specs 056-058 settle — see the TODO in
    face_grid.py.

    Assertion: at least one of the cluster's faces has a resolvable crop
    file under ``run_dir/crops/`` using the hardcoded suffix pattern.
    """
    from sim_bench.db.face_clustering.cluster_analysis_repo import (
        ClusterAnalysisRepoConfig,
        ClusterAnalysisRepository,
    )
    from face_cluster.views.cluster_analysis import ClusterAnalysisService

    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=no_op_merge_run_dir))
    service = ClusterAnalysisService(repo)
    rows = service.list_clusters()
    if not rows:
        pytest.skip("Synthetic fixture has no real clusters — face-grid contract is vacuous.")

    view = service.compute_detail(cluster_id=rows[0].cluster_id)
    assert view.faces, "ClusterView.faces is empty — face grid would render nothing."

    crops_dir = no_op_merge_run_dir / "crops"
    existing = [
        crops_dir / f"face_{f.face_id:04d}_aligned.jpg"
        for f in view.faces
        if (crops_dir / f"face_{f.face_id:04d}_aligned.jpg").is_file()
    ]
    assert existing, (
        f"No crop file resolved under {crops_dir} for any face in cluster "
        f"{rows[0].cluster_id}. Tried pattern face_{{id:04d}}_aligned.jpg. "
        f"Sample face_ids: {[f.face_id for f in view.faces[:3]]}. "
        f"Dir contents: {sorted(p.name for p in crops_dir.iterdir())[:5] if crops_dir.exists() else 'MISSING'}"
    )


@pytest.mark.slow
def test_cluster_analysis_metrics_actually_render(no_op_merge_run_dir):
    """SIGHTING-079 regression: previously the tab reached
    "Analysing cluster…" and never advanced — AsyncHandle started a
    background thread but Streamlit doesn't poll it. After the sync
    rewrite, the metrics MUST render (5 st.metric widgets from
    render_cluster_metrics).

    This is the assertion that would have caught SIGHTING-079 before it
    shipped. The earlier no-crash test passed even on the broken UI.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(MAIN_PY)
    at.session_state["v2_last_run_dir"] = str(no_op_merge_run_dir)
    at.run(timeout=60)

    n_metrics = len(at.metric)
    # render_cluster_metrics emits 5 metrics (Faces / Diameter / Avg intra-dist
    # / Exemplars / Outliers); render_cluster_debug emits 4 more inside an
    # expander (which AppTest still counts). Total >= 5 means metrics rendered.
    assert n_metrics >= 5, (
        f"Cluster Analysis tab metrics did not render: {n_metrics} metrics on page. "
        f"Previously SIGHTING-079: stuck on 'Analysing cluster…' caption forever."
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


@pytest.mark.slow
def test_main_page_renders_without_duplicate_widget_keys_with_recluster_available(
    no_op_merge_run_dir, isolate_action_log_db
):
    """Regression for the spec-063 duplicate widget-key crash.

    Run tab and Recluster tab both render the FCParams editor. Both tab
    bodies execute on every Streamlit script run. Without per-tab key
    namespacing, both tried to create st.slider(key="v2_K") in the same
    script run -> StreamlitDuplicateElementKey -> entire page crashes
    (every tab fails to render its h2).

    The earlier "no run loaded" smoke test missed this because, with an
    empty action_log, render_run_picker returns None and the Recluster
    tab early-exits BEFORE reaching render_field. This test seeds an
    action_log row so the picker proceeds, exercising the code path that
    triggered the bug in the browser.
    """
    from face_cluster.run_history_db import init_table
    from streamlit.testing.v1 import AppTest
    from tests.face_clustering.views._seed import insert_action

    init_table(db_path=isolate_action_log_db)
    insert_action(
        isolate_action_log_db,
        status="complete",
        output_dir=str(no_op_merge_run_dir),
        source_album="spec-063-duplicate-key-regression",
        producer="fc_app_v2",
        action_type="fc_app_v2_run",
    )

    at = AppTest.from_file(MAIN_PY)
    at.run(timeout=60)

    exceptions = [str(e.value) for e in at.exception]
    duplicate_key_errors = [m for m in exceptions if "multiple elements with the same" in m]
    assert not duplicate_key_errors, (
        "Page raised StreamlitDuplicateElementKey — two tabs share a widget "
        "key. Fix by passing key_prefix= to render_field/render_group from "
        "the second tab. Offending messages:\n  " + "\n  ".join(duplicate_key_errors)
    )
    assert not exceptions, (
        f"Page raised {len(exceptions)} other exception(s):\n  "
        + "\n  ".join(exceptions)
    )

    slider_keys = [s.key for s in at.slider]
    assert len(slider_keys) == len(set(slider_keys)), (
        f"Duplicate slider keys present: {sorted(slider_keys)}"
    )
