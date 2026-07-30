"""Typed search helper for the action_log history table.

.. deprecated:: spec-043
    Use :class:`face_cluster.repositories.RunHistoryRepository` instead.
    This module's free functions remain for backward compatibility with
    legacy callers; removal is tracked as a follow-up after the
    spec-043 burn-in period.

Usage (deprecated):
    from face_cluster.run_history import search, HistoryFilters, RunRow
    rows = search(HistoryFilters(album="Noa2_5"), db_path=None)

Replacement (preferred):
    from face_cluster.repositories import (
        RunHistoryRepository, RunHistoryRepoConfig, RunHistoryCriteria,
    )
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=None))
    rows = repo.find(RunHistoryCriteria(album="Noa2_5"))
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from face_cluster.run_history_db import get_db_path, init_table

# spec-043: emit once per process. Visible in test output / CI logs;
# doesn't flood production. Future PR adding a new caller surfaces this.
warnings.warn(
    "face_cluster.run_history is deprecated; use "
    "face_cluster.repositories.RunHistoryRepository instead. "
    "Tracked for removal after the spec-043 burn-in period.",
    DeprecationWarning,
    stacklevel=2,
)

_TABLE = "action_log"
_UNKNOWN_ALBUM = "(unknown)"


@dataclass
class RunRow:
    id: int
    action_type: str
    status: str
    started_at: str
    ended_at: Optional[str]
    duration_s: Optional[float]
    run_id: Optional[str]
    source_dir: Optional[str]
    output_dir: Optional[str]
    album: Optional[str]
    n_faces: Optional[int]
    n_clusters: Optional[int]
    n_noise: Optional[int]
    log_file: Optional[str]
    # spec-013 fields
    source_album: str = _UNKNOWN_ALBUM
    run_name: Optional[str] = None
    parent_run_id: Optional[int] = None
    run_kind: Optional[str] = None
    comment: Optional[str] = None
    config_json: Optional[str] = None
    n_core: Optional[int] = None
    # spec-043: payload_json exposed so the Repository's find() can populate
    # it for callers that need the raw action payload (e.g., HistoryService
    # formatting non-pipeline actions). Pre-spec-043 callers ignored it.
    payload_json: Optional[str] = None
    producer: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_orm(cls, model) -> "RunRow":
        """Build a RunRow from an ActionLog ORM instance.

        Single source of truth for the ORM → dataclass mapping. Column
        names are taken from ``ActionLog.__table__.columns`` so any
        column add/rename is caught by ``test_runrow_matches_action_log``.

        Preserves the legacy ``source_album`` → ``_UNKNOWN_ALBUM``
        fallback when the column is NULL/empty.
        """
        values = {c.name: getattr(model, c.name) for c in model.__table__.columns}
        if not values.get("source_album"):
            values["source_album"] = _UNKNOWN_ALBUM
        return cls(**values)

    @property
    def display_album(self) -> str:
        return self.source_album or _UNKNOWN_ALBUM

    @property
    def config(self) -> dict:
        return json.loads(self.config_json) if self.config_json else {}

    @property
    def payload(self) -> dict:
        """Parsed payload_json. Empty dict when None/missing/malformed."""
        if not self.payload_json:
            return {}
        try:
            return json.loads(self.payload_json)
        except Exception:
            return {}


@dataclass
class HistoryFilters:
    album: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    text: Optional[str] = None
    limit: int = 500


def _connect(db_path: Optional[Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_run_row(row: sqlite3.Row) -> RunRow:
    d = dict(row)
    return RunRow(
        id=d["id"],
        action_type=d["action_type"],
        status=d["status"],
        started_at=d["started_at"],
        ended_at=d.get("ended_at"),
        duration_s=d.get("duration_s"),
        run_id=d.get("run_id"),
        source_dir=d.get("source_dir"),
        output_dir=d.get("output_dir"),
        album=d.get("album"),
        n_faces=d.get("n_faces"),
        n_clusters=d.get("n_clusters"),
        n_noise=d.get("n_noise"),
        log_file=d.get("log_file"),
        source_album=d.get("source_album") or _UNKNOWN_ALBUM,
        run_name=d.get("run_name"),
        parent_run_id=d.get("parent_run_id"),
        run_kind=d.get("run_kind"),
        comment=d.get("comment"),
        config_json=d.get("config_json"),
        n_core=d.get("n_core"),
        payload_json=d.get("payload_json"),
        producer=d.get("producer"),
        error=d.get("error"),
    )


def search(
    filters: HistoryFilters,
    db_path: Optional[Path] = None,
) -> list[RunRow]:
    """Return action_log rows matching filters, newest-first."""
    init_table(db_path)
    clauses: list[str] = ["status != 'reserved'"]
    params: list = []

    if filters.album:
        clauses.append("source_album = ?")
        params.append(filters.album)
    if filters.date_from:
        clauses.append("started_at >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to:
        # include the full day
        clauses.append("started_at < date(?, '+1 day')")
        params.append(filters.date_to.isoformat())
    if filters.text:
        like = f"%{filters.text}%"
        clauses.append("(source_album LIKE ? OR run_name LIKE ? OR comment LIKE ?)")
        params.extend([like, like, like])

    where = " AND ".join(clauses)
    sql = f"SELECT * FROM {_TABLE} WHERE {where} ORDER BY started_at DESC LIMIT ?"
    params.append(filters.limit)

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_run_row(r) for r in rows]


def distinct_albums(db_path: Optional[Path] = None) -> list[str]:
    """Return sorted list of distinct non-NULL source_album values."""
    init_table(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT source_album FROM {_TABLE} "
            f"WHERE source_album IS NOT NULL AND source_album != '' "
            f"AND status != 'reserved' ORDER BY source_album"
        ).fetchall()
    return [r[0] for r in rows]


def get_run_by_id(
    run_id: int,
    db_path: Optional[Path] = None,
) -> Optional[RunRow]:
    """Fetch a single row by primary key."""
    init_table(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE id=?", (run_id,)
        ).fetchone()
    return _row_to_run_row(row) if row else None
