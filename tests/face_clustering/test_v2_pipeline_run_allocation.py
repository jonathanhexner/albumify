"""spec-050 Phase 2 — pipeline persists allocator outputs into action_log.

Runs ``run_v2_pipeline`` with an empty source dir so the producer chain
short-circuits with ``success=False`` before any ML loads. That's enough
to verify the action_log row gets the caller-supplied ``run_id``,
``output_dir``, ``source_album``, and ``producer`` fields exactly.

The full producer/clustering chain is covered by
``test_fc_app_v2_e2e.py`` and (when green) the equivalence suite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.face_clustering_v2.pipeline import run_v2_pipeline
from face_cluster._paths import default_db_path
from face_cluster.run_layout import allocate_run_dir


@pytest.fixture
def isolated_action_log_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default action_log DB at a tmp file so the test doesn't
    write into the user's real ~/.sim_bench/sim_bench.db."""
    fake_db = tmp_path / "isolated_sim_bench.db"
    monkeypatch.setattr(
        "face_cluster._paths.default_db_path",
        lambda: fake_db,
    )
    # _resolve_db_path in run_history_repo imports _paths.default_db_path
    # at call time, so the monkeypatch above is enough.
    return fake_db


def _read_latest_action_log_row(db_path: Path) -> dict:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM action_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None, "no action_log rows"
    return dict(row)


def test_pipeline_records_caller_supplied_run_id_album_output_dir(
    tmp_path: Path,
    isolated_action_log_db: Path,
) -> None:
    src = tmp_path / "src_empty"
    src.mkdir()
    runs_base = tmp_path / "runs"
    run_dir, run_id = allocate_run_dir(runs_base, album_slug="phase2_test")

    result = run_v2_pipeline(
        src_dir=src,
        run_dir=run_dir,
        run_id=run_id,
        album="phase2_test",
    )
    # Empty src → success=False, but action_log row must still be written.
    assert result.success is False

    row = _read_latest_action_log_row(isolated_action_log_db)
    assert row["run_id"] == run_id
    assert row["run_id"] == run_dir.name
    assert row["output_dir"] == str(run_dir)
    assert row["source_album"] == "phase2_test"
    assert row["producer"] == "fc_app_v2"
    assert row["action_type"] == "fc_app_v2_run"
