"""SIGHTING-080 regression: HistoryService.get_run_detail must mark v5 runs
(face_clustering.db at top level) as loadable.

Before the 2026-05-29 fix, ``_REQUIRED_ARTIFACTS = ('faces.csv', 'clusters.csv',
'embeddings.npy')`` was a hardcoded legacy-CSV list. Every v2 run failed the
artifact check even when the v5 sqlite DB was present and
``load_pipeline_result`` could open it. Symptom in the v2 History tab:
*"Run is incomplete (status: complete). Cannot load — required artifacts ...
are missing or status is not 'complete'."* (the user-reported contradiction).

The fix replaces the hardcoded list with ``_run_dir_has_loadable_artifacts``,
which matches the three layouts ``load_pipeline_result`` actually handles:
  - v5 (spec-040):  face_clustering.db at top level
  - v4 transitional: _v4/face_clustering.db
  - legacy CSV:     faces.csv + clusters.csv + embeddings.npy
"""
from __future__ import annotations

from pathlib import Path

from face_cluster.views.history import _run_dir_has_loadable_artifacts


def test_loadable_false_for_nonexistent_dir(tmp_path):
    assert _run_dir_has_loadable_artifacts(tmp_path / "does_not_exist") is False


def test_loadable_false_for_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run_dir_has_loadable_artifacts(empty) is False


def test_loadable_true_for_v5_layout(tmp_path):
    """The exact shape spec-040 / fc_app_v2 produces."""
    run = tmp_path / "v5_run"
    run.mkdir()
    (run / "face_clustering.db").touch()
    # embeddings.npy / pipeline_run.json / crops also exist but only the DB
    # is required by load_pipeline_result for v5.
    assert _run_dir_has_loadable_artifacts(run) is True


def test_loadable_true_for_v4_transitional_layout(tmp_path):
    """spec-030 transitional: DB nested under _v4/."""
    run = tmp_path / "v4_run"
    (run / "_v4").mkdir(parents=True)
    (run / "_v4" / "face_clustering.db").touch()
    assert _run_dir_has_loadable_artifacts(run) is True


def test_loadable_true_for_legacy_csv_layout(tmp_path):
    """Pre-spec-030 CSV trio. Still openable by load_pipeline_result's
    legacy fallback."""
    run = tmp_path / "legacy_csv_run"
    run.mkdir()
    (run / "faces.csv").touch()
    (run / "clusters.csv").touch()
    (run / "embeddings.npy").touch()
    assert _run_dir_has_loadable_artifacts(run) is True


def test_loadable_false_when_legacy_csv_partial(tmp_path):
    """Two of three legacy CSVs is not enough — load_pipeline_result needs all three."""
    run = tmp_path / "partial_csv"
    run.mkdir()
    (run / "faces.csv").touch()
    (run / "embeddings.npy").touch()
    # clusters.csv missing
    assert _run_dir_has_loadable_artifacts(run) is False


def test_loadable_false_when_db_path_is_a_dir(tmp_path):
    """Defensive: face_clustering.db as a directory (pathological) is not loadable."""
    weird = tmp_path / "weird"
    weird.mkdir()
    (weird / "face_clustering.db").mkdir()  # not a file
    assert _run_dir_has_loadable_artifacts(weird) is False


# ---------------------------------------------------------------------------
# SIGHTING-089 regression: v5 run_metadata table populates RunSummary
# (legacy pipeline_run.json doesn't carry the keys; History panel blank)
# ---------------------------------------------------------------------------

def test_summary_from_run_metadata_populates_fields_for_v5_run(tmp_path):
    """SIGHTING-089: v5 runs write only 9 keys to pipeline_run.json (no
    `summary` / `stages` / `merge_metadata` blocks). The History detail
    panel used to render blank for every v2 run.

    Fix: read summary stats from the `run_metadata` DB table via RunStore.
    This test builds a v5-shaped synthetic run dir + asserts get_run_detail
    returns a populated RunSummary (n_faces / n_clusters_base / etc.).
    """
    from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
        _build_synthetic_run_dir,
    )
    from face_cluster.views.history import HistoryService
    from face_cluster.run_history_db import init_table
    from tests.face_clustering.views._seed import insert_action

    run_dir = _build_synthetic_run_dir(tmp_path)

    # Seed an action_log row pointing at the v5 run dir.
    action_db = tmp_path / "sim_bench.db"
    init_table(db_path=action_db)
    rid = insert_action(
        action_db,
        status="complete",
        output_dir=str(run_dir),
        source_album="SIGHTING-089-regression",
        producer="fc_app_v2",
        action_type="fc_app_v2_run",
    )

    # Monkey-patch the default DB path so HistoryService reads from our tmp DB.
    import face_cluster._paths as _paths
    orig = _paths.default_db_path
    _paths.default_db_path = lambda: action_db
    try:
        detail = HistoryService().get_run_detail(rid)
    finally:
        _paths.default_db_path = orig

    # Before the SIGHTING-089 fix, every field of RunSummary was None for v5
    # runs (because pipeline_run.json didn't carry the legacy keys). After
    # the fix, RunStore.metadata() supplies the numbers.
    assert detail.summary is not None, "RunSummary should not be None for v5 run"
    assert detail.summary.n_faces == 32, (
        f"n_faces empty — SIGHTING-089 regressed. Got: {detail.summary}"
    )
    assert detail.summary.n_clusters_base == 3
    assert detail.summary.n_clusters_merged == 3
    assert detail.summary.merge_count == 0
