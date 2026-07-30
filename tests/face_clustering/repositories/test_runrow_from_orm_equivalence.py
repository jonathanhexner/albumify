"""spec-048 Phase 4 — `RunRow.from_orm` is byte-identical to the
deleted ``_to_run_row`` helper across the 24-row golden fixture.

Runs the new (derived) mapping against every committed real-data row
and compares against a frozen copy of the legacy explicit mapping.
Once spec-048 is merged, the frozen copy here is the only remaining
trace of the old code — kept as the equivalence oracle.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

from face_cluster.repositories._engine import create_engine_for_path
from face_cluster.repositories._session import make_sessionmaker, session_scope
from face_cluster.repositories.models.action_log import ActionLog
from face_cluster.run_history import RunRow, _UNKNOWN_ALBUM


def _legacy_to_run_row(m: ActionLog) -> RunRow:
    """Frozen copy of the spec-046 ``_to_run_row`` for equivalence testing."""
    return RunRow(
        id=m.id,
        action_type=m.action_type,
        status=m.status,
        started_at=m.started_at,
        ended_at=m.ended_at,
        duration_s=m.duration_s,
        run_id=m.run_id,
        source_dir=m.source_dir,
        output_dir=m.output_dir,
        album=m.album,
        n_faces=m.n_faces,
        n_clusters=m.n_clusters,
        n_noise=m.n_noise,
        log_file=m.log_file,
        source_album=m.source_album or _UNKNOWN_ALBUM,
        run_name=m.run_name,
        parent_run_id=m.parent_run_id,
        run_kind=m.run_kind,
        comment=m.comment,
        config_json=m.config_json,
        n_core=m.n_core,
        payload_json=m.payload_json,
        producer=m.producer,
        error=m.error,
    )


def test_from_orm_byte_identical_to_legacy_for_every_golden_row(
    golden_run_history_db_copy: Path,
) -> None:
    engine = create_engine_for_path(golden_run_history_db_copy)
    sm = make_sessionmaker(engine)
    try:
        with session_scope(sm) as session:
            models = list(session.query(ActionLog).order_by(ActionLog.id).all())
            assert len(models) >= 20, f"expected ≥20 golden rows, got {len(models)}"
            for m in models:
                new = RunRow.from_orm(m)
                legacy = _legacy_to_run_row(m)
                assert asdict(new) == asdict(legacy), (
                    f"row id={m.id}: drift between RunRow.from_orm and legacy mapping"
                )
    finally:
        engine.dispose()


def test_from_orm_applies_unknown_album_fallback_when_null() -> None:
    model = ActionLog(
        action_type="test",
        status="complete",
        started_at="2026-05-28T10:00:00+00:00",
        source_album=None,
    )
    row = RunRow.from_orm(model)
    assert row.source_album == _UNKNOWN_ALBUM


def test_from_orm_preserves_existing_source_album() -> None:
    model = ActionLog(
        action_type="test",
        status="complete",
        started_at="2026-05-28T10:00:00+00:00",
        source_album="Budapest",
    )
    row = RunRow.from_orm(model)
    assert row.source_album == "Budapest"
