"""T007 — DB migration test: spec-013 columns added idempotently."""
import sqlite3
from pathlib import Path

import pytest

from face_cluster import run_history_db


_NEW_COLUMNS = {
    "source_album", "run_name", "parent_run_id", "run_kind",
    "comment", "config_json", "n_core",
}


def _column_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    names = {row[1] for row in conn.execute("PRAGMA table_info(action_log)")}
    conn.close()
    return names


def test_migration_adds_new_columns(tmp_path):
    """init_table() on a fresh DB must create all spec-013 columns."""
    db = tmp_path / "test.db"
    run_history_db.init_table(db)
    cols = _column_names(db)
    assert _NEW_COLUMNS.issubset(cols), f"Missing columns: {_NEW_COLUMNS - cols}"


def test_migration_is_idempotent(tmp_path):
    """Calling init_table() twice must not raise and must not duplicate columns."""
    db = tmp_path / "test.db"
    run_history_db.init_table(db)
    run_history_db.init_table(db)  # second call
    cols = _column_names(db)
    assert _NEW_COLUMNS.issubset(cols)


def test_migration_preserves_existing_data(tmp_path):
    """Rows inserted before the migration columns were added are unaffected."""
    db = tmp_path / "test.db"

    # Simulate a pre-013 DB: create the table without new columns
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_s REAL,
            error TEXT,
            run_id TEXT,
            source_dir TEXT,
            output_dir TEXT,
            album TEXT,
            n_faces INTEGER,
            n_clusters INTEGER,
            n_noise INTEGER,
            log_file TEXT,
            payload_json TEXT
        );
        INSERT INTO action_log (action_type, status, started_at, run_id)
        VALUES ('pipeline_run', 'complete', '2026-01-01T00:00:00+00:00', 'old-run-1');
    """)
    conn.commit()
    conn.close()

    # Now run migration
    run_history_db.init_table(db)

    cols = _column_names(db)
    assert _NEW_COLUMNS.issubset(cols)

    # Pre-existing row must still be present
    conn2 = sqlite3.connect(str(db))
    row = conn2.execute("SELECT run_id, source_album FROM action_log WHERE run_id='old-run-1'").fetchone()
    conn2.close()
    assert row is not None
    assert row[0] == "old-run-1"
    assert row[1] is None  # source_album defaulted to NULL


def test_update_comment_persists(tmp_path):
    """update_comment() must write and retrieve a comment."""
    db = tmp_path / "test.db"
    action_id = run_history_db.start_action("test", payload={}, db_path=db)
    run_history_db.update_comment(action_id, "my note", db_path=db)
    row = run_history_db.get_action(action_id, db_path=db)
    assert row["comment"] == "my note"


def test_update_comment_enforces_length_limit(tmp_path):
    """Comments longer than 2048 characters must raise ValueError."""
    db = tmp_path / "test.db"
    action_id = run_history_db.start_action("test", payload={}, db_path=db)
    long_comment = "x" * 2049
    with pytest.raises(ValueError, match="2048"):
        run_history_db.update_comment(action_id, long_comment, db_path=db)
