"""Row factories for view-service unit tests.

Seeds rows into a test-owned SQLite DB directly via SQL (bypasses the
``start_action`` / ``complete_action`` helpers so tests can construct
arbitrary states — failed runs, runs with parents, runs missing fields,
etc.).

Convention: ``_make_*`` returns a dict; ``insert_*`` writes it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    """Return a Row-factory connection to the test DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def make_action(
    *,
    id: Optional[int] = None,
    action_type: str = "fc_app_v2_run",
    status: str = "complete",
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    duration_s: Optional[float] = None,
    error: Optional[str] = None,
    run_id: Optional[str] = None,
    source_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    album: Optional[str] = None,
    n_faces: Optional[int] = None,
    n_clusters: Optional[int] = None,
    n_noise: Optional[int] = None,
    log_file: Optional[str] = None,
    payload_json: Optional[str] = None,
    source_album: Optional[str] = "TestAlbum",
    run_name: Optional[str] = None,
    parent_run_id: Optional[int] = None,
    run_kind: Optional[str] = "full",
    comment: Optional[str] = None,
    config_json: Optional[str] = None,
    n_core: Optional[int] = None,
    producer: Optional[str] = "fc_app_v2",
) -> dict[str, Any]:
    """Build a kwargs dict suitable for inserting into action_log.

    All v5-schema fields explicit. Sensible defaults so tests only
    override what they care about.
    """
    if started_at is None:
        started_at = datetime.now(timezone.utc)
    started_iso = _iso(started_at)
    ended_iso = _iso(ended_at) if ended_at else None
    return {
        "id": id,
        "action_type": action_type,
        "status": status,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "duration_s": duration_s,
        "error": error,
        "run_id": run_id,
        "source_dir": source_dir,
        "output_dir": output_dir,
        "album": album,
        "n_faces": n_faces,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "log_file": log_file,
        "payload_json": payload_json,
        "source_album": source_album,
        "run_name": run_name,
        "parent_run_id": parent_run_id,
        "run_kind": run_kind,
        "comment": comment,
        "config_json": config_json,
        "n_core": n_core,
        "producer": producer,
    }


def insert_action(db_path: Path, **fields: Any) -> int:
    """Insert one action_log row built from ``make_action`` kwargs; return id.

    Example::

        rid = insert_action(db, source_album="Budapest", n_clusters=33)
    """
    row = make_action(**fields)
    cols = [k for k, v in row.items() if k != "id"]
    placeholders = ", ".join("?" for _ in cols)
    values = [row[c] for c in cols]
    sql = f"INSERT INTO action_log ({', '.join(cols)}) VALUES ({placeholders})"
    with _connect(db_path) as conn:
        cur = conn.execute(sql, values)
        conn.commit()
        return cur.lastrowid


def seed_runs(
    db_path: Path,
    *,
    albums: tuple[str, ...] = ("Budapest", "Paris"),
    runs_per_album: int = 3,
    base_time: Optional[datetime] = None,
    base_kwargs: Optional[dict[str, Any]] = None,
) -> list[int]:
    """Insert a small grid of runs across albums and dates; return inserted ids.

    Default produces ``len(albums) * runs_per_album`` rows, spaced one
    hour apart in started_at, all completed. Override per-call via
    ``base_kwargs``.
    """
    if base_time is None:
        base_time = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    extra = base_kwargs or {}
    ids: list[int] = []
    offset = 0
    for album in albums:
        for i in range(runs_per_album):
            ids.append(
                insert_action(
                    db_path,
                    source_album=album,
                    started_at=base_time + timedelta(hours=offset),
                    run_name=f"{album}-run-{i}",
                    n_faces=100 + offset,
                    n_clusters=5 + i,
                    n_core=80 + offset,
                    **extra,
                )
            )
            offset += 1
    return ids
