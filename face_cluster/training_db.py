"""Central SQLite storage for merge training data and trained model registry.

Uses raw sqlite3 (stdlib) so face_cluster stays independent of sim_bench.api ORM.
Database: ~/.sim_bench/sim_bench.db (shared with the main application).

Tables
------
merge_training_data:
    id              INTEGER PK AUTOINCREMENT
    run_id          TEXT    NOT NULL            -- output folder name
    album_path      TEXT                        -- source image directory
    cluster_a       INTEGER NOT NULL            -- smaller cluster ID
    cluster_b       INTEGER NOT NULL            -- larger cluster ID
    label           INTEGER                     -- 1=approve, 0=reject, NULL=unlabeled
    feature_version INTEGER NOT NULL            -- FeatureComputer.VERSION
    features_json   TEXT    NOT NULL            -- ClusterPairFeatures as JSON
    output_dir      TEXT                        -- full path to run output dir (for crop lookup)
    exemplar_ids    TEXT                        -- JSON: {"a": [12,45,7], "b": [3,88,21]}
    saved_at        TEXT    NOT NULL            -- ISO-8601
    UNIQUE(run_id, cluster_a, cluster_b)

trained_models:
    id              INTEGER PK AUTOINCREMENT
    name            TEXT    NOT NULL UNIQUE     -- e.g. "merge_lr_20260417_143000"
    model_type      TEXT    NOT NULL            -- logistic_regression | xgboost | mlp
    model_path      TEXT    NOT NULL            -- absolute path to .pkl file
    feature_version INTEGER NOT NULL
    n_samples       INTEGER
    n_train         INTEGER
    n_test          INTEGER
    accuracy        REAL
    f1              REAL
    auc_roc         REAL
    created_at      TEXT    NOT NULL            -- ISO-8601
    metadata_json   TEXT                        -- full metrics + hyperparams + feature_groups
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TABLE        = "merge_training_data"
_MODELS_TABLE = "trained_models"

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    album_path      TEXT,
    cluster_a       INTEGER NOT NULL,
    cluster_b       INTEGER NOT NULL,
    label           INTEGER,
    feature_version INTEGER NOT NULL,
    features_json   TEXT    NOT NULL,
    output_dir      TEXT,
    exemplar_ids    TEXT,
    saved_at        TEXT    NOT NULL,
    source          TEXT,
    verified        INTEGER DEFAULT 0,
    UNIQUE(run_id, cluster_a, cluster_b)
)
"""

_COLS = [
    "run_id", "album_path", "cluster_a", "cluster_b", "label",
    "feature_version", "features_json", "output_dir", "exemplar_ids", "saved_at",
]

# Extended columns including source/verified — used by INSERT OR IGNORE in label verification.
_COLS_EXT = _COLS + ["source", "verified"]


def get_db_path() -> Path:
    """Return path to ~/.sim_bench/sim_bench.db."""
    from face_cluster._paths import default_db_path
    return default_db_path()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_add_source_verified(conn: sqlite3.Connection) -> None:
    """Add source and verified columns to an existing table (idempotent)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()}
    if "source" not in existing:
        conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN source TEXT DEFAULT NULL")
    if "verified" not in existing:
        conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN verified INTEGER DEFAULT 0")


def init_training_table(db_path: Optional[Path] = None) -> None:
    """Create merge_training_data table if it does not exist (idempotent)."""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_SQL)
        _migrate_add_source_verified(conn)
        conn.commit()


def upsert_training_samples(
    samples: List[Dict],
    db_path: Optional[Path] = None,
) -> int:
    """Insert or replace training samples. Returns count written.

    Each sample dict must contain: run_id, cluster_a, cluster_b, feature_version,
    features_json, saved_at. Optional: album_path, label, output_dir, exemplar_ids.
    Existing rows with the same (run_id, cluster_a, cluster_b) are overwritten.
    """
    if not samples:
        return 0
    init_training_table(db_path)
    placeholders = ", ".join(["?"] * len(_COLS))
    sql = f"INSERT OR REPLACE INTO {_TABLE} ({', '.join(_COLS)}) VALUES ({placeholders})"
    rows = [tuple(s.get(c) for c in _COLS) for s in samples]
    with _connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
    logger.info("Upserted %d training samples to %s", len(rows), db_path or get_db_path())
    return len(rows)


def load_training_data(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all rows. features_json is expanded into individual columns.

    Returns an empty DataFrame if the table has no rows.
    """
    init_training_table(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM {_TABLE}").fetchall()
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        r = dict(row)
        raw_json = r.pop("features_json", None)
        if raw_json:
            try:
                r.update(json.loads(raw_json))
            except (json.JSONDecodeError, TypeError):
                pass
        records.append(r)
    return pd.DataFrame(records)


def training_data_summary(db_path: Optional[Path] = None) -> Dict:
    """Return aggregate stats without loading feature vectors."""
    init_training_table(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(f"""
            SELECT
                COUNT(*)                                        AS total_rows,
                COUNT(DISTINCT run_id)                          AS n_runs,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)     AS n_approved,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END)     AS n_rejected,
                SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END)  AS n_unlabeled,
                MAX(feature_version)                            AS latest_feature_version
            FROM {_TABLE}
        """).fetchone()
    return dict(row) if row else {}


def update_label(
    run_id: str,
    cluster_a: int,
    cluster_b: int,
    new_label: Optional[int],
    db_path: Optional[Path] = None,
) -> bool:
    """Flip the label for a single (run_id, cluster_a, cluster_b) row.

    Args:
        new_label: 1 (approve), 0 (reject), or None (unlabeled).

    Returns:
        True if a row was updated, False if the row was not found.
    """
    init_training_table(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE {_TABLE} SET label = ? WHERE run_id = ? AND cluster_a = ? AND cluster_b = ?",
            (new_label, run_id, cluster_a, cluster_b),
        )
        conn.commit()
    updated = cur.rowcount > 0
    if updated:
        logger.info("Label updated: run=%s pair=(%d,%d) -> %s", run_id, cluster_a, cluster_b, new_label)
    return updated


# ---------------------------------------------------------------------------
# Label verification helpers
# ---------------------------------------------------------------------------

def insert_heuristic_samples(
    samples: List[Dict],
    db_path: Optional[Path] = None,
) -> int:
    """INSERT OR IGNORE heuristic-labelled samples (never overwrites human labels).

    Like upsert_training_samples but uses INSERT OR IGNORE and includes source/verified.
    Each sample must have the standard _COLS fields plus optionally source and verified.
    """
    if not samples:
        return 0
    init_training_table(db_path)
    placeholders = ", ".join(["?"] * len(_COLS_EXT))
    sql = f"INSERT OR IGNORE INTO {_TABLE} ({', '.join(_COLS_EXT)}) VALUES ({placeholders})"
    rows = [tuple(s.get(c) for c in _COLS_EXT) for s in samples]
    with _connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)


def save_human_label(
    run_id: str,
    cluster_a: int,
    cluster_b: int,
    label: Optional[int],
    verified: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Update a pair with a human decision.

    Args:
        label: 1=merge, 0=reject, None=ignore (excluded from training).
        verified: Whether the user explicitly reviewed this pair.

    Returns True if a row was updated.
    """
    init_training_table(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""UPDATE {_TABLE}
                SET label=?, source='human', verified=?
                WHERE run_id=? AND cluster_a=? AND cluster_b=?""",
            (label, 1 if verified else 0, run_id, cluster_a, cluster_b),
        )
        conn.commit()
    updated = cur.rowcount > 0
    if updated:
        logger.debug("Human label saved: run=%s pair=(%d,%d) label=%s verified=%s",
                     run_id, cluster_a, cluster_b, label, verified)
    return updated


def get_labels_for_run(
    run_id: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return label/source/verified rows for a run (without feature vectors)."""
    init_training_table(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cluster_a, cluster_b, label, source, verified "
            f"FROM {_TABLE} WHERE run_id=?",
            (run_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["cluster_a", "cluster_b", "label", "source", "verified"])
    return pd.DataFrame([dict(r) for r in rows])


def label_summary_by_run(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Per-run counts of merge/reject/ignore/unverified for the summary panel."""
    init_training_table(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(f"""
            SELECT
                run_id,
                SUM(CASE WHEN label=1 AND (source='heuristic' OR source IS NULL) THEN 1 ELSE 0 END) AS heuristic_merge,
                SUM(CASE WHEN label=0 AND (source='heuristic' OR source IS NULL) THEN 1 ELSE 0 END) AS heuristic_reject,
                SUM(CASE WHEN label=1 AND source='human' THEN 1 ELSE 0 END) AS human_merge,
                SUM(CASE WHEN label=0 AND source='human' THEN 1 ELSE 0 END) AS human_reject,
                SUM(CASE WHEN label IS NULL AND source='human' THEN 1 ELSE 0 END) AS human_ignore,
                SUM(CASE WHEN verified=0 OR verified IS NULL THEN 1 ELSE 0 END) AS unverified
            FROM {_TABLE}
            GROUP BY run_id
        """).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Trained model registry
# ---------------------------------------------------------------------------

_CREATE_MODELS_SQL = f"""
CREATE TABLE IF NOT EXISTS {_MODELS_TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    model_type      TEXT    NOT NULL,
    model_path      TEXT    NOT NULL,
    feature_version INTEGER NOT NULL,
    n_samples       INTEGER,
    n_train         INTEGER,
    n_test          INTEGER,
    accuracy        REAL,
    f1              REAL,
    auc_roc         REAL,
    created_at      TEXT    NOT NULL,
    metadata_json   TEXT,
    UNIQUE(name)
)
"""

_MODEL_COLS = [
    "name", "model_type", "model_path", "feature_version",
    "n_samples", "n_train", "n_test",
    "accuracy", "f1", "auc_roc",
    "created_at", "metadata_json",
]


def _init_models_table(db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(_CREATE_MODELS_SQL)
        conn.commit()


def save_model_record(record: Dict, db_path: Optional[Path] = None) -> None:
    """Insert or replace a model record in the trained_models table.

    Expected keys in record: name, model_type, model_path, feature_version,
    n_samples, n_train, n_test, accuracy, f1, auc_roc, created_at, metadata_json.
    """
    _init_models_table(db_path)
    placeholders = ", ".join(["?"] * len(_MODEL_COLS))
    sql = (
        f"INSERT OR REPLACE INTO {_MODELS_TABLE} "
        f"({', '.join(_MODEL_COLS)}) VALUES ({placeholders})"
    )
    row = tuple(record.get(c) for c in _MODEL_COLS)
    with _connect(db_path) as conn:
        conn.execute(sql, row)
        conn.commit()
    logger.info("Model record saved: %s", record.get("name"))


def load_model_records(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return all rows from trained_models ordered by created_at DESC."""
    _init_models_table(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM {_MODELS_TABLE} ORDER BY created_at DESC"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])
