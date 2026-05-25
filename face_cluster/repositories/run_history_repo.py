"""spec-043 — RunHistoryRepository: typed persistence access to action_log.

Replaces the ~600 LOC of module-level free functions in
``face_cluster/run_history.py`` and ``face_cluster/run_history_db.py``
with one Repository class. Pattern: see
``docs/architecture/architecture_standards.md`` §B0.

Three rules followed verbatim:

1. The persistence layer is a class, not free functions.
2. The constructor takes a typed Config dataclass; adding a new option
   is non-breaking for existing callers.
3. Query methods take composable typed criteria (``RunHistoryCriteria``);
   adding a new filter axis is a new field, not a new method.

Services (``face_cluster.views.history.HistoryService``) compose this
via constructor injection. Legacy free functions still work (with a
``DeprecationWarning``); removal is a follow-up spec after burn-in.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from face_cluster.repositories._errors import NotFoundError, ValidationError
from face_cluster.run_history import RunRow, _row_to_run_row

logger = logging.getLogger(__name__)

_TABLE = "action_log"
_COMMENT_MAX_LEN = 2048
_TEXT_FIELD_ALLOWLIST: frozenset[str] = frozenset({
    "source_album", "run_name", "comment", "album", "run_id",
})


# ---------------------------------------------------------------------------
# Config + Criteria — typed boundaries per B0 / B4 of the standards
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunHistoryRepoConfig:
    """Configuration for :class:`RunHistoryRepository`.

    Every field has a default. Adding a new field is non-breaking for
    existing callers — they keep whatever values they had before.

    Attributes:
        db_path: path to a SQLite file with the ``action_log`` schema.
            When ``None``, defaults to ``~/.sim_bench/sim_bench.db``.
        auto_migrate: when ``True`` (default), the Repository runs the
            idempotent schema migration (``init_table``) on first use.
            Set ``False`` in tests that pre-seed a custom schema state.
        read_only: when ``True``, mutation methods raise
            :class:`ValidationError` rather than touching the DB.
            Forward-looking — not enforced yet by all methods.
        log_queries: when ``True``, emit DEBUG log lines on every SQL
            execution. Off by default to keep logs quiet.
        connection_timeout_s: SQLite ``timeout`` for connection
            establishment. Defaults to 5 seconds.
    """
    db_path: Optional[Path] = None
    auto_migrate: bool = True
    read_only: bool = False
    log_queries: bool = False
    connection_timeout_s: float = 5.0


@dataclass(frozen=True, slots=True)
class RunHistoryCriteria:
    """Composable filter for run-history queries.

    All fields are optional. Empty/None means "no filter on that axis";
    combination is AND, not OR. Adding a new filter axis = adding a
    field here; the Repository's ``find()`` / ``find_one()`` / ``count()``
    methods stay stable.

    Attributes:
        ids: restrict to these action_log primary keys. None = no id filter.
        parent_run_id: restrict to rows whose ``parent_run_id`` matches.
        album: exact match on ``source_album``.
        status: exact match on ``status`` (e.g. "complete", "failed").
        producer: exact match on ``producer`` (e.g. "fc_app_v2").
        action_types: include rows whose ``action_type`` is one of these.
        date_from: rows started on/after this date.
        date_to: rows started on/before this date (inclusive — full end day).
        text: substring match across ``text_fields``, case-insensitive.
        text_fields: which columns ``text`` searches. Allowlisted at the
            Repository to prevent SQL injection of arbitrary column names.
        limit: max rows returned. Default 500 to match legacy ``search``.
        offset: pagination offset. Default 0.
        order_by: ordering of returned rows. Today only ``started_at``
            direction is supported; extend if needed.
    """
    ids: Optional[List[int]] = None
    parent_run_id: Optional[int] = None
    album: Optional[str] = None
    status: Optional[str] = None
    producer: Optional[str] = None
    action_types: Optional[List[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    text: Optional[str] = None
    text_fields: tuple[str, ...] = ("source_album", "run_name", "comment")
    limit: int = 500
    offset: int = 0
    order_by: Literal["started_at_desc", "started_at_asc"] = "started_at_desc"


# ---------------------------------------------------------------------------
# Schema migration (mirrors face_cluster.run_history_db)
# ---------------------------------------------------------------------------

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

# Columns added via ALTER TABLE (idempotent migration); mirrors
# face_cluster.run_history_db._ALTER_COLUMNS.
_ALTER_COLUMNS: list[tuple[str, str]] = [
    ("source_album",  "TEXT"),
    ("run_name",      "TEXT"),
    ("parent_run_id", "INTEGER"),
    ("run_kind",      "TEXT"),
    ("comment",       "TEXT"),
    ("config_json",   "TEXT"),
    ("n_core",        "INTEGER"),
    ("producer",      "TEXT"),
]

# Fields the legacy code calls "hot" — top-level columns that can be
# populated by start_action / complete_action result_fields.
_HOT_FIELDS = (
    "run_id", "source_dir", "output_dir", "album",
    "n_faces", "n_clusters", "n_noise", "log_file",
    "source_album", "run_name", "parent_run_id", "run_kind",
    "comment", "config_json", "n_core", "producer",
)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class RunHistoryRepository:
    """Read + write access to the ``action_log`` table. Thin SQL wrapper.

    Owns: a connection path to a SQLite DB (default
    ``~/.sim_bench/sim_bench.db``). Knows the schema; knows nothing
    about views, services, Streamlit, or business logic.

    Construction takes a :class:`RunHistoryRepoConfig` (or ``None`` for
    defaults). Query methods take :class:`RunHistoryCriteria`; mutation
    methods take named arguments per the B3 service-vocabulary
    standard.

    Test isolation: pass an explicit ``db_path`` via the Config — each
    Repository instance owns its own connection path. No global
    monkeypatching required.
    """

    def __init__(self, config: Optional[RunHistoryRepoConfig] = None):
        """Initialize with a Config (or None for defaults).

        Side effects: runs the idempotent schema migration on first
        connect when ``config.auto_migrate`` is True (the default).
        """
        self._config = config or RunHistoryRepoConfig()
        self._migrated = False

    # ------------------------------------------------------------------
    # Connection management (private)
    # ------------------------------------------------------------------

    def _db_path(self) -> Path:
        """Resolve the configured DB path, falling back to the default."""
        if self._config.db_path is not None:
            return self._config.db_path
        # Use the global default. Module-attribute lookup so a future
        # arch-test-friendly refactor of the default-path resolver works.
        from face_cluster import run_history_db
        return run_history_db.get_db_path()

    def _connect(self) -> sqlite3.Connection:
        """Open a row-factory SQLite connection. Runs migration on first call."""
        conn = sqlite3.connect(
            str(self._db_path()),
            timeout=self._config.connection_timeout_s,
        )
        conn.row_factory = sqlite3.Row
        if self._config.auto_migrate and not self._migrated:
            self._migrate(conn)
            self._migrated = True
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent schema migration. Creates table + adds missing columns."""
        conn.executescript(_CREATE_SQL)
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
        conn.commit()

    def _check_writable(self) -> None:
        """Raise ValidationError if the Repository is in read-only mode."""
        if self._config.read_only:
            raise ValidationError(
                "RunHistoryRepository is configured read_only=True; "
                "mutations are disabled.",
                user_message="This run history is read-only.",
            )

    # ------------------------------------------------------------------
    # Composable queries
    # ------------------------------------------------------------------

    def find(self, criteria: RunHistoryCriteria) -> List[RunRow]:
        """Return rows matching ``criteria``, ordered + paginated.

        Args:
            criteria: typed filter. Empty criteria returns all rows
                (up to ``criteria.limit``).

        Returns:
            list of :class:`RunRow`. Empty when no rows match.

        Raises:
            :class:`ValidationError`: when criteria fields are
                self-inconsistent (e.g. ``date_from > date_to``).
        """
        where, params = self._build_where(criteria)
        order = self._order_clause(criteria.order_by)
        sql = (
            f"SELECT * FROM {_TABLE} WHERE {where} "
            f"{order} LIMIT ? OFFSET ?"
        )
        params = (*params, criteria.limit, criteria.offset)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_run_row(r) for r in rows]

    def find_one(self, criteria: RunHistoryCriteria) -> Optional[RunRow]:
        """First row matching ``criteria``, or None.

        Convenience wrapper that issues a ``LIMIT 1`` query rather than
        slicing the full result of :meth:`find`.
        """
        scoped = self._with_limit(criteria, 1)
        results = self.find(scoped)
        return results[0] if results else None

    def count(self, criteria: RunHistoryCriteria) -> int:
        """Total rows matching ``criteria`` without fetching them.

        Args:
            criteria: typed filter. ``limit`` / ``offset`` ignored.

        Returns:
            integer count.
        """
        where, params = self._build_where(criteria)
        sql = f"SELECT COUNT(*) FROM {_TABLE} WHERE {where}"
        with self._connect() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def distinct_albums(self) -> List[str]:
        """Sorted list of distinct non-empty ``source_album`` values."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT source_album FROM {_TABLE} "
                f"WHERE source_album IS NOT NULL AND source_album != '' "
                f"AND status != 'reserved' "
                f"ORDER BY source_album"
            ).fetchall()
        return [r[0] for r in rows]

    def get_by_id(self, run_id: int) -> Optional[RunRow]:
        """Single-id lookup. Returns None when ``run_id`` doesn't exist."""
        return self.find_one(RunHistoryCriteria(ids=[run_id]))

    # ------------------------------------------------------------------
    # Mutations — named verbs per B3 of the standards
    # ------------------------------------------------------------------

    def start_action(
        self,
        action_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a new ``action_log`` row in 'running' status.

        Args:
            action_type: short tag (e.g. "fc_app_v2_run", "merge_apply").
            payload: dict serialized to ``payload_json``. May contain
                "hot field" keys (``run_id``, ``source_album``,
                ``producer``, etc.) which are also written to top-level
                columns for indexed queries.

        Returns:
            The newly-inserted row's ``id`` (action_id).

        Side effects: INSERTs one row.
        """
        self._check_writable()
        payload = payload or {}
        hot = {k: payload.get(k) for k in _HOT_FIELDS}
        with self._connect() as conn:
            cur = conn.execute(
                f"""INSERT INTO {_TABLE}
                    (action_type, status, started_at,
                     run_id, source_dir, output_dir, album,
                     n_faces, n_clusters, n_noise, log_file,
                     source_album, run_name, parent_run_id, run_kind,
                     comment, config_json, n_core, producer,
                     payload_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    action_type, "running", _now_iso(),
                    hot["run_id"], hot["source_dir"], hot["output_dir"], hot["album"],
                    hot["n_faces"], hot["n_clusters"], hot["n_noise"], hot["log_file"],
                    hot["source_album"], hot["run_name"], hot["parent_run_id"], hot["run_kind"],
                    hot["comment"], hot["config_json"], hot["n_core"], hot["producer"],
                    json.dumps(payload),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def complete_action(
        self,
        action_id: int,
        *,
        result_fields: Optional[Dict[str, Any]] = None,
        payload_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark an action complete; merge ``result_fields`` into the row.

        Args:
            action_id: action_log primary key.
            result_fields: dict whose keys may be any of the hot
                fields (``n_faces``, ``n_clusters``, ``producer``, etc.).
                Each non-None value is written via COALESCE so
                existing values aren't overwritten with NULL.
            payload_update: dict merged (top-level) into ``payload_json``.

        Raises:
            :class:`NotFoundError`: when ``action_id`` does not exist.

        Side effects: UPDATEs status, ended_at, duration, hot fields,
        and payload_json.
        """
        self._check_writable()
        result_fields = result_fields or {}
        ended = _now_iso()
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT started_at, payload_json FROM {_TABLE} WHERE id=?",
                (action_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in {_TABLE}",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            started = datetime.fromisoformat(row["started_at"])
            duration = (datetime.fromisoformat(ended) - started).total_seconds()
            payload = json.loads(row["payload_json"] or "{}")
            if payload_update:
                payload.update(payload_update)
            hot = {k: result_fields.get(k) for k in _HOT_FIELDS}
            conn.execute(
                f"""UPDATE {_TABLE} SET
                    status='complete', ended_at=?, duration_s=?,
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
                    comment=COALESCE(?,comment),
                    config_json=COALESCE(?,config_json),
                    n_core=COALESCE(?,n_core),
                    producer=COALESCE(?,producer),
                    payload_json=?
                    WHERE id=?""",
                (
                    ended, duration,
                    hot["run_id"], hot["source_dir"], hot["output_dir"], hot["album"],
                    hot["n_faces"], hot["n_clusters"], hot["n_noise"], hot["log_file"],
                    hot["source_album"], hot["run_name"], hot["parent_run_id"], hot["run_kind"],
                    hot["comment"], hot["config_json"], hot["n_core"], hot["producer"],
                    json.dumps(payload),
                    action_id,
                ),
            )
            conn.commit()

    def fail_action(
        self,
        action_id: int,
        error_text: str,
        *,
        payload_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark an action failed with an error message.

        Args:
            action_id: action_log primary key.
            error_text: developer-facing error string.
            payload_update: optional dict merged into ``payload_json``.

        Raises:
            :class:`NotFoundError`: when ``action_id`` does not exist.

        Side effects: UPDATEs status='failed', ended_at, duration, error.
        """
        self._check_writable()
        ended = _now_iso()
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT started_at, payload_json FROM {_TABLE} WHERE id=?",
                (action_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in {_TABLE}",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            started = datetime.fromisoformat(row["started_at"])
            duration = (datetime.fromisoformat(ended) - started).total_seconds()
            payload = json.loads(row["payload_json"] or "{}")
            if payload_update:
                payload.update(payload_update)
            conn.execute(
                f"""UPDATE {_TABLE} SET
                    status='failed', ended_at=?, duration_s=?,
                    error=?, payload_json=?
                    WHERE id=?""",
                (ended, duration, error_text, json.dumps(payload), action_id),
            )
            conn.commit()

    def update_comment(self, action_id: int, comment: str) -> None:
        """Set the ``comment`` field on an action.

        Args:
            action_id: action_log primary key.
            comment: free text, max 2048 characters.

        Raises:
            :class:`ValidationError`: when ``comment`` exceeds 2048 chars.
            :class:`NotFoundError`: when ``action_id`` does not exist.

        Side effects: UPDATEs ``comment``. Idempotent — same comment
        twice produces the same final state.
        """
        self._check_writable()
        if len(comment) > _COMMENT_MAX_LEN:
            raise ValidationError(
                f"Comment exceeds {_COMMENT_MAX_LEN} characters "
                f"(got {len(comment)})",
                user_message=f"Comments are limited to {_COMMENT_MAX_LEN} characters.",
            )
        with self._connect() as conn:
            # Existence check first so we can raise NotFoundError instead of
            # silently succeeding with zero rows affected (legacy behavior).
            row = conn.execute(
                f"SELECT 1 FROM {_TABLE} WHERE id=?", (action_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in {_TABLE}",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            conn.execute(
                f"UPDATE {_TABLE} SET comment=? WHERE id=?",
                (comment, action_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # WHERE-clause builder (private)
    # ------------------------------------------------------------------

    def _build_where(
        self, criteria: RunHistoryCriteria
    ) -> tuple[str, tuple]:
        """Translate a RunHistoryCriteria into a (where_sql, params) pair.

        Raises:
            :class:`ValidationError`: when criteria fields are
                self-inconsistent (date_from > date_to) or refer to
                unknown text_fields.
        """
        # Cross-field validation
        if (
            criteria.date_from is not None
            and criteria.date_to is not None
            and criteria.date_from > criteria.date_to
        ):
            raise ValidationError(
                f"date_from ({criteria.date_from}) > date_to ({criteria.date_to})",
                user_message="Invalid date range — 'from' is after 'to'.",
            )
        unknown_text_fields = set(criteria.text_fields) - _TEXT_FIELD_ALLOWLIST
        if unknown_text_fields:
            raise ValidationError(
                f"Unknown text_fields: {sorted(unknown_text_fields)}. "
                f"Allowed: {sorted(_TEXT_FIELD_ALLOWLIST)}",
                user_message="Internal error: invalid search field.",
            )

        clauses: List[str] = ["status != 'reserved'"]
        params: List[Any] = []

        if criteria.ids:
            placeholders = ",".join("?" for _ in criteria.ids)
            clauses.append(f"id IN ({placeholders})")
            params.extend(criteria.ids)
        if criteria.parent_run_id is not None:
            clauses.append("parent_run_id = ?")
            params.append(criteria.parent_run_id)
        if criteria.album:
            clauses.append("source_album = ?")
            params.append(criteria.album)
        if criteria.status:
            clauses.append("status = ?")
            params.append(criteria.status)
        if criteria.producer:
            clauses.append("producer = ?")
            params.append(criteria.producer)
        if criteria.action_types:
            placeholders = ",".join("?" for _ in criteria.action_types)
            clauses.append(f"action_type IN ({placeholders})")
            params.extend(criteria.action_types)
        if criteria.date_from is not None:
            clauses.append("started_at >= ?")
            params.append(criteria.date_from.isoformat())
        if criteria.date_to is not None:
            # Inclusive end-of-day, matching legacy run_history.search.
            clauses.append("started_at < date(?, '+1 day')")
            params.append(criteria.date_to.isoformat())
        if criteria.text:
            like = f"%{criteria.text}%"
            text_clauses = " OR ".join(f"{col} LIKE ?" for col in criteria.text_fields)
            clauses.append(f"({text_clauses})")
            params.extend([like] * len(criteria.text_fields))

        return " AND ".join(clauses), tuple(params)

    def _order_clause(self, order_by: str) -> str:
        """Translate the typed order_by literal to a SQL ORDER BY clause."""
        return {
            "started_at_desc": "ORDER BY started_at DESC",
            "started_at_asc": "ORDER BY started_at ASC",
        }[order_by]

    @staticmethod
    def _with_limit(criteria: RunHistoryCriteria, limit: int) -> RunHistoryCriteria:
        """Return a copy of ``criteria`` with a different limit."""
        from dataclasses import replace
        return replace(criteria, limit=limit)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return UTC now as an ISO-8601 string. Matches legacy behavior."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RunHistoryRepoConfig",
    "RunHistoryCriteria",
    "RunHistoryRepository",
]
