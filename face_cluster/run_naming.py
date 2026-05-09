"""Atomic output-directory allocation for face-clustering runs.

Guarantees unique per-run output paths even under concurrent calls within
the same session.  Uses a dedicated `run_dir_reservations` table with a
UNIQUE constraint on `path` so INSERT OR IGNORE is safe.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from face_cluster import run_history_db

logger = logging.getLogger(__name__)

_RESERVATIONS_TABLE = "run_dir_reservations"
_CREATE_RESERVATIONS = f"""
CREATE TABLE IF NOT EXISTS {_RESERVATIONS_TABLE} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    source_album TEXT,
    kind        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class RunDirSpec:
    """Parameters that determine an output directory name."""
    source_album: str   # base name of the source image folder
    kind: str           # 'recluster' | 'remerge' | 'base' | 'manual_merge'
    label: str = ""     # optional human suffix


def allocate_run_dir(
    spec: RunDirSpec,
    results_root: Path,
    db_path: Optional[Path] = None,
) -> Path:
    """Return a unique, reserved output directory path.

    Scans `results_root / spec.source_album / <kind>_N` starting at N=1,
    incrementing until neither the directory exists on disk NOR a
    reservation exists for that path.

    The returned path is atomically reserved.  The directory itself is NOT
    created by this function — the caller must do that before starting work.
    """
    _init_reservations(db_path)
    album_dir = results_root / spec.source_album
    n = 1
    while True:
        candidate = album_dir / f"{spec.kind}_{n}"
        if not candidate.exists() and _try_reserve(candidate, spec, db_path):
            logger.debug("allocate_run_dir: reserved %s", candidate)
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_conn(db_path: Optional[Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or run_history_db.get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _init_reservations(db_path: Optional[Path]) -> None:
    with _db_conn(db_path) as conn:
        conn.executescript(_CREATE_RESERVATIONS)
        conn.commit()


def _try_reserve(
    candidate: Path,
    spec: RunDirSpec,
    db_path: Optional[Path],
) -> bool:
    """Atomically insert a reservation. Returns True on success."""
    with _db_conn(db_path) as conn:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO {_RESERVATIONS_TABLE} (path, source_album, kind) VALUES (?,?,?)",
            (str(candidate), spec.source_album, spec.kind),
        )
        conn.commit()
        return cur.rowcount > 0
