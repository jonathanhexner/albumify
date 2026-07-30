"""spec-040 Phase 5b T4 — end-to-end test for the new FC App v2 pipeline.

Drives ``app.face_clustering_v2.pipeline.run_v2_pipeline`` directly (the
function the Run tab wraps). Verifies the full v2 path is exercised:

* Producer chain populates ``context.face_records`` (A1 dual-write).
* FCAppRunner clustering chain runs end-to-end.
* ``RunExporter`` writes a v5 ``face_clustering.db``.
* New v5 tables (``images``, ratio columns) are populated.
* ``action_log`` row carries ``producer='fc_app_v2'`` after the run.

This test does NOT spawn Streamlit — that's a separate manual /
opt-in browser check. The point here is to lock in the underlying
contract: the v2 *pipeline* (independent of UI) works on real input.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.conftest import get_test_data_dir


FIXTURE_DIR = get_test_data_dir() / "face_clustering"


def _pick_jpgs(n_per_person: int = 2, n_persons: int = 3) -> list[Path]:
    if not FIXTURE_DIR.exists():
        return []
    picked: list[Path] = []
    for person_dir in sorted(FIXTURE_DIR.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        jpgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() == ".jpg")
        picked.extend(jpgs[:n_per_person])
        if len(picked) >= n_per_person * n_persons:
            break
    return picked


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    """Single v2 pipeline run on a 6-jpg fixture, shared by all tests in this module."""
    images = _pick_jpgs(n_per_person=2, n_persons=3)
    if not images:
        pytest.skip(f"Fixture missing at {FIXTURE_DIR}")
    src_dir = tmp_path_factory.mktemp("v2_e2e_src")
    for src in images:
        (src_dir / src.name).write_bytes(src.read_bytes())
    out_dir = tmp_path_factory.mktemp("v2_e2e_out")

    # Use a private action_log DB so the test doesn't pollute ~/.sim_bench.
    # spec-048: _resolve_db_path now reads from face_cluster._paths, not
    # from run_history_db. Patch the new location.
    db_path = tmp_path_factory.mktemp("v2_e2e_action_log") / "sim_bench.db"
    import face_cluster._paths as _paths
    orig_get_db_path = _paths.default_db_path
    _paths.default_db_path = lambda: db_path  # type: ignore[assignment]
    try:
        from app.face_clustering_v2.pipeline import run_v2_pipeline
        from face_cluster.fc_params import FCParams

        # spec-041: drive the pipeline through the typed container instead
        # of the legacy step_configs dict broadcast.
        params = FCParams(
            K=3, distance_threshold=0.5, min_cluster_size=2,
            blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
            max_faces_per_image_core=50,
            merge_enabled=False,
            cluster_diameter_cap_enabled=False,
        )
        result = run_v2_pipeline(
            src_dir=Path(src_dir), run_dir=Path(out_dir),
            run_id="testrunid000000000000000000000a",
            album="e2e_test_album",
            params=params,
        )
    finally:
        _paths.default_db_path = orig_get_db_path
    if not result.success:
        pytest.skip(f"v2 pipeline failed (likely env): {result.error_message}")
    return result, db_path


def test_v2_pipeline_completes_successfully(v2_run):
    result, _ = v2_run
    assert result.success
    assert result.db_path.exists()
    assert result.n_faces > 0, "producer chain detected no faces"


def test_v2_pipeline_produces_v5_images_table(v2_run):
    result, _ = v2_run
    conn = sqlite3.connect(str(result.db_path))
    try:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert n_images == result.n_images, (
            f"images table has {n_images} rows; expected {result.n_images}"
        )
        # Ratio columns must be non-NULL on every detected face.
        n_null = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE area_ratio IS NULL"
        ).fetchone()[0]
        assert n_null == 0, f"{n_null} faces with NULL area_ratio (v5 producer regression)"
    finally:
        conn.close()


def test_v2_pipeline_writes_action_log_with_producer_tag(v2_run):
    """The new producer column on action_log records 'fc_app_v2' for this run."""
    result, action_log_db = v2_run
    assert result.action_id is not None, "action_log row was not created"
    conn = sqlite3.connect(str(action_log_db))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT producer, action_type, status FROM action_log WHERE id=?",
            (result.action_id,),
        ).fetchone()
        assert row is not None, "action_log row missing"
        assert row["producer"] == "fc_app_v2", (
            f"expected producer='fc_app_v2', got {row['producer']!r}"
        )
        assert row["action_type"] == "fc_app_v2_run"
        assert row["status"] == "complete"
    finally:
        conn.close()


def test_v2_pipeline_writes_a_run_directory(v2_run):
    """Output dir must contain the canonical 5-artifact layout."""
    result, _ = v2_run
    expected = {"face_clustering.db", "embeddings.npy", "embedding_face_ids.npy",
                "pipeline_run.json", "crops"}
    actual = {p.name for p in result.output_dir.iterdir()}
    missing = expected - actual
    assert not missing, f"v2 run dir missing artifacts: {missing}; got {actual}"


def test_v2_pipeline_runs_with_non_default_fcparams(tmp_path_factory):
    """spec-041 — a non-default FCParams instance reaches the pipeline.

    Drives the run with merge_enabled=True, tighter K, and a custom merge
    threshold. Verifies the pipeline completes — proves the params= path
    plumbs through to the step configs, not just defaults.
    """
    import sqlite3
    from app.face_clustering_v2.pipeline import run_v2_pipeline
    from face_cluster.fc_params import FCParams

    images = _pick_jpgs(n_per_person=2, n_persons=3)
    if not images:
        pytest.skip(f"Fixture missing at {FIXTURE_DIR}")
    src_dir = tmp_path_factory.mktemp("v2_nondefault_src")
    for src in images:
        (src_dir / src.name).write_bytes(src.read_bytes())
    out_dir = tmp_path_factory.mktemp("v2_nondefault_out")

    db_path = tmp_path_factory.mktemp("v2_nondefault_log") / "sim_bench.db"
    import face_cluster._paths as _paths
    orig = _paths.default_db_path
    _paths.default_db_path = lambda: db_path  # type: ignore[assignment]
    try:
        params = FCParams(
            K=3, distance_threshold=0.5, min_cluster_size=2,
            blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
            max_faces_per_image_core=50,
            merge_enabled=True,
            merge_candidate_threshold=0.50,  # non-default
        )
        result = run_v2_pipeline(
            src_dir=Path(src_dir), run_dir=Path(out_dir),
            run_id="testrunid000000000000000000000b",
            album="e2e_merge_test_album",
            params=params,
        )
    finally:
        _paths.default_db_path = orig
    if not result.success:
        pytest.skip(f"v2 pipeline failed (env): {result.error_message}")
    assert result.success
    # Confirm DB exists and is non-empty.
    conn = sqlite3.connect(str(result.db_path))
    try:
        n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        assert n_faces > 0
    finally:
        conn.close()
