"""Persistent action log for every user-initiated face-clustering operation.

All actions (pipeline runs, reclusters, merge applies, profile saves, ML training,
model loads) are written to a single `action_log` table in the shared SQLite DB at
~/.sim_bench/sim_bench.db.

Usage — context manager (recommended):
    from face_cluster.run_history_db import log_action

    with log_action("recluster", payload={"source_dir": src, "config": cfg}) as action_id:
        result = pipeline.recluster(src, out)
    # complete_action is called automatically with no extra fields.
    # On exception, fail_action is called with the traceback.

Usage — manual:
    action_id = start_action("merge_apply", payload={...})
    ...
    complete_action(action_id, {"n_clusters": 12, "n_noise": 3})
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_TABLE = "action_log"

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type     TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'running',
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    duration_s      REAL,
    error           TEXT,
    run_id          TEXT,
    source_dir      TEXT,
    output_dir      TEXT,
    album           TEXT,
    n_faces         INTEGER,
    n_clusters      INTEGER,
    n_noise         INTEGER,
    log_file        TEXT,
    payload_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_log_started
    ON {_TABLE}(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_log_type
    ON {_TABLE}(action_type, started_at DESC);
"""

_HOT_FIELDS = frozenset({
    "run_id", "source_dir", "output_dir", "album",
    "n_faces", "n_clusters", "n_noise", "log_file",
    # spec-013 fields
    "source_album", "run_name", "parent_run_id", "run_kind", "comment",
    "config_json", "n_core",
    # spec-040 Phase 4 field — distinguishes which app produced a run.
    "producer",
})

_COMMENT_MAX_LEN = 2048

# Spec-013 columns added via ALTER TABLE (idempotent migration)
_ALTER_COLUMNS: list[tuple[str, str]] = [
    ("source_album",  "TEXT"),
    ("run_name",      "TEXT"),
    ("parent_run_id", "INTEGER"),
    ("run_kind",      "TEXT"),
    ("comment",       "TEXT"),
    ("config_json",   "TEXT"),
    ("n_core",        "INTEGER"),
    # spec-040 Phase 4 — distinguishes runs by producer app
    # (albumify | fc_app | fc_app_v2). Idempotent add via the same migration
    # path as the spec-013 columns; reads from existing rows return NULL.
    ("producer",      "TEXT"),
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    db_dir = Path.home() / ".sim_bench"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sim_bench.db"


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_013(conn: sqlite3.Connection) -> None:
    """Idempotent migration: add spec-013 columns if absent."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({_TABLE})")}
    for col, col_type in _ALTER_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {col} {col_type}")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_action_log_source_album "
        f"ON {_TABLE}(source_album)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_action_log_comment "
        f"ON {_TABLE}(comment)"
    )


def init_table(db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_CREATE_SQL)
        _migrate_013(conn)
        conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def start_action(
    action_type: str,
    payload: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a 'running' row.  Returns the new action_id."""
    init_table(db_path)
    payload = payload or {}
    hot = {k: payload.get(k) for k in _HOT_FIELDS}
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""INSERT INTO {_TABLE}
                (action_type, status, started_at,
                 run_id, source_dir, output_dir, album,
                 n_faces, n_clusters, n_noise, log_file,
                 source_album, run_name, parent_run_id, run_kind,
                 config_json, n_core,
                 producer,
                 payload_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                action_type, "running", _now_iso(),
                hot.get("run_id"), hot.get("source_dir"),
                hot.get("output_dir"), hot.get("album"),
                hot.get("n_faces"), hot.get("n_clusters"),
                hot.get("n_noise"), hot.get("log_file"),
                hot.get("source_album"), hot.get("run_name"),
                hot.get("parent_run_id"), hot.get("run_kind"),
                hot.get("config_json"), hot.get("n_core"),
                hot.get("producer"),
                json.dumps(payload),
            ),
        )
        conn.commit()
        return cur.lastrowid


def complete_action(
    action_id: int,
    result_fields: Optional[Dict[str, Any]] = None,
    payload_update: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Mark an action as complete, writing hot fields + merged payload."""
    result_fields = result_fields or {}
    ended = _now_iso()

    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT started_at, payload_json FROM {_TABLE} WHERE id=?", (action_id,)
        ).fetchone()
        if row is None:
            logger.warning("complete_action: action_id %d not found", action_id)
            return

        started = datetime.fromisoformat(row["started_at"])
        ended_dt = datetime.fromisoformat(ended)
        duration = (ended_dt - started).total_seconds()

        payload = json.loads(row["payload_json"] or "{}")
        if payload_update:
            payload.update(payload_update)

        hot = {k: result_fields.get(k) for k in _HOT_FIELDS}
        conn.execute(
            f"""UPDATE {_TABLE} SET
                status=?, ended_at=?, duration_s=?,
                run_id=COALESCE(?,run_id),
                source_dir=COALESCE(?,source_dir),
                output_dir=COALESCE(?,output_dir),
                album=COALESCE(?,album),
                n_faces=COALESCE(?,n_faces),
                n_clusters=COALESCE(?,n_clusters),
                n_noise=COALESCE(?,n_noise),
                log_file=COALESCE(?,log_file),
                source_album=COALESCE(?,source_album),
                run_name=COALESCE(?,run_name),
                parent_run_id=COALESCE(?,parent_run_id),
                run_kind=COALESCE(?,run_kind),
                config_json=COALESCE(?,config_json),
                n_core=COALESCE(?,n_core),
                producer=COALESCE(?,producer),
                payload_json=?
                WHERE id=?""",
            (
                "complete", ended, duration,
                hot.get("run_id"), hot.get("source_dir"),
                hot.get("output_dir"), hot.get("album"),
                hot.get("n_faces"), hot.get("n_clusters"),
                hot.get("n_noise"), hot.get("log_file"),
                hot.get("source_album"), hot.get("run_name"),
                hot.get("parent_run_id"), hot.get("run_kind"),
                hot.get("config_json"), hot.get("n_core"),
                hot.get("producer"),
                json.dumps(payload),
                action_id,
            ),
        )
        conn.commit()
    logger.debug("Action %d complete (%.1fs)", action_id, duration)


def fail_action(
    action_id: int,
    error_text: str,
    payload_update: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Mark an action as failed."""
    ended = _now_iso()
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT started_at, payload_json FROM {_TABLE} WHERE id=?", (action_id,)
        ).fetchone()
        if row is None:
            logger.warning("fail_action: action_id %d not found", action_id)
            return

        started = datetime.fromisoformat(row["started_at"])
        duration = (datetime.fromisoformat(ended) - started).total_seconds()

        payload = json.loads(row["payload_json"] or "{}")
        if payload_update:
            payload.update(payload_update)

        conn.execute(
            f"""UPDATE {_TABLE} SET
                status='failed', ended_at=?, duration_s=?, error=?,
                payload_json=?
                WHERE id=?""",
            (ended, duration, error_text, json.dumps(payload), action_id),
        )
        conn.commit()
    logger.debug("Action %d failed: %s", action_id, error_text[:80])


def update_comment(
    action_id: int,
    comment: str,
    db_path: Optional[Path] = None,
) -> None:
    """Persist a user comment on a run row.  Enforces 2048-char limit."""
    if len(comment) > _COMMENT_MAX_LEN:
        raise ValueError(f"Comment exceeds {_COMMENT_MAX_LEN} characters")
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET comment=? WHERE id=?", (comment, action_id)
        )
        conn.commit()


def list_actions(
    types: Optional[List[str]] = None,
    limit: int = 200,
    db_path: Optional[Path] = None,
) -> List[Dict]:
    """Return recent actions sorted newest-first."""
    init_table(db_path)
    with _connect(db_path) as conn:
        if types:
            placeholders = ",".join("?" * len(types))
            rows = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE action_type IN ({placeholders})"
                f" ORDER BY started_at DESC LIMIT ?",
                (*types, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {_TABLE} ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_action(
    action_id: int,
    db_path: Optional[Path] = None,
) -> Optional[Dict]:
    init_table(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE id=?", (action_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def log_action(
    action_type: str,
    payload: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Generator[int, None, None]:
    """Context manager that auto-completes or fails an action row.

    Yields the action_id so callers can enrich it before the block exits.
    On success: calls complete_action() with no extra result_fields.
    On exception: calls fail_action() then re-raises.

    For richer completion data (n_clusters etc.) use the manual API instead.
    """
    action_id = start_action(action_type, payload=payload, db_path=db_path)
    yielded_state: Dict[str, Any] = {}
    try:
        yield action_id
        complete_action(action_id, result_fields=yielded_state, db_path=db_path)
    except Exception as exc:
        import traceback
        fail_action(action_id, traceback.format_exc(), db_path=db_path)
        raise


def purge_stale_runs(db_path: Optional[Path] = None) -> int:
    """Remove DB entries whose output_dir no longer exists or lacks key files.

    Also marks stuck 'running' entries (no thread is actually running) as 'failed'.
    Returns the number of rows deleted.
    """
    init_table(db_path)
    to_delete: List[int] = []
    to_fail: List[int] = []

    with _connect(db_path) as conn:
        rows = conn.execute(f"SELECT id, output_dir, status FROM {_TABLE}").fetchall()
        for row in rows:
            rid = row["id"]
            out = row["output_dir"] or ""
            status = row["status"] or ""

            out_path = Path(out)
            # Resolve relative paths against CWD (they were stored that way)
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path

            dir_exists = out_path.is_dir()
            has_key_file = dir_exists and (
                (out_path / "faces.csv").exists()
                or (out_path / "pipeline_run.json").exists()
                or (out_path / "cluster_result.npz").exists()
            )

            if not dir_exists:
                to_delete.append(rid)
            elif not has_key_file and status in ("complete", "running"):
                # Directory exists but has no useful output
                to_delete.append(rid)
            elif status == "running":
                # Stale 'running' entry — no thread is active
                to_fail.append(rid)

        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE id IN ({placeholders})", to_delete
            )
        if to_fail:
            placeholders = ",".join("?" * len(to_fail))
            conn.execute(
                f"UPDATE {_TABLE} SET status='failed', error='stale running entry' "
                f"WHERE id IN ({placeholders})", to_fail
            )
        conn.commit()

    n = len(to_delete)
    if n:
        logger.info("Purged %d stale run history entries", n)
    if to_fail:
        logger.info("Marked %d stale 'running' entries as failed", len(to_fail))
    return n


def upsert_run(
    run_id: str,
    output_dir: str,
    action_type: str,
    fields: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> None:
    """Insert a completed run row if (run_id, output_dir) is not already present.

    Used by the migration script only.
    """
    init_table(db_path)
    with _connect(db_path) as conn:
        exists = conn.execute(
            f"SELECT 1 FROM {_TABLE} WHERE run_id=? AND output_dir=?",
            (run_id, output_dir),
        ).fetchone()
        if exists:
            return

        payload = fields.pop("payload_json_extra", {})
        conn.execute(
            f"""INSERT INTO {_TABLE}
                (action_type, status, started_at, ended_at, duration_s, error,
                 run_id, source_dir, output_dir, album,
                 n_faces, n_clusters, n_noise, log_file, payload_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                action_type,
                fields.get("status", "complete"),
                fields.get("started_at", _now_iso()),
                fields.get("ended_at"),
                fields.get("duration_s"),
                fields.get("error"),
                run_id,
                fields.get("source_dir"),
                output_dir,
                fields.get("album"),
                fields.get("n_faces"),
                fields.get("n_clusters"),
                fields.get("n_noise"),
                fields.get("log_file"),
                json.dumps(payload),
            ),
        )
        conn.commit()
