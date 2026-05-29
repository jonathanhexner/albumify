"""spec-046 / spec-048 — RunHistoryRepository (SQLAlchemy-backed).

Read + write access to the action_log table. Schema is owned by Alembic
(see ``alembic/versions/``). Public API and return types match the
spec-043 contract.

Construction:
    repo = RunHistoryRepository()                                    # default DB
    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=...))   # custom DB

Each method opens its own short SQLAlchemy session via
``session_scope``; engine + sessionmaker are owned by the instance
(spec-048 removed the module-level engine cache). Future multi-call
transactions: add a ``from_session(session)`` constructor when a real
need materialises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from sqlalchemy import func, or_, select

from face_cluster.repositories._engine import create_engine_for_path
from face_cluster.repositories._errors import NotFoundError, ValidationError
from face_cluster.repositories._schema import ensure_schema
from face_cluster.repositories._session import make_sessionmaker, session_scope
from face_cluster.repositories.models.action_log import ActionLog
from face_cluster.run_history import RunRow

_COMMENT_MAX_LEN = 2048

_TEXT_FIELD_ALLOWLIST: frozenset[str] = frozenset({
    "source_album", "run_name", "comment", "album", "run_id",
})



# ---------------------------------------------------------------------------
# Criteria — typed query input. Kept from spec-043 for caller compatibility.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunHistoryCriteria:
    """Composable filter for run-history queries. AND across fields."""
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


# Backwards-compat shim. Pre-spec-046 callers constructed
# RunHistoryRepository(RunHistoryRepoConfig(db_path=...)); new code just
# passes db_path directly. Kept so the burn-in window is non-disruptive.
@dataclass(frozen=True, slots=True)
class RunHistoryRepoConfig:
    db_path: Optional[Path] = None
    read_only: bool = False


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Optional[Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    from face_cluster._paths import default_db_path
    return default_db_path()


class RunHistoryRepository:
    """Read + write access to the action_log table.

    Construction:
        RunHistoryRepository()                                    # default DB
        RunHistoryRepository(RunHistoryRepoConfig(db_path=...))   # explicit path

    Each method opens its own short SQLAlchemy session.
    """

    def __init__(self, config: Optional[RunHistoryRepoConfig] = None):
        self._read_only = False
        db_path: Optional[Path] = None
        if config is not None:
            db_path = config.db_path
            self._read_only = config.read_only
        self._db_path = _resolve_db_path(db_path)
        ensure_schema(self._db_path)
        self._engine = create_engine_for_path(self._db_path)
        self._sm = make_sessionmaker(self._engine)

    def _check_writable(self) -> None:
        if self._read_only:
            raise ValidationError(
                "Repository is read_only=True; mutations are disabled.",
                user_message="This run history is read-only.",
            )

    # ------------------------------------------------------------------ reads

    def find(self, criteria: RunHistoryCriteria) -> List[RunRow]:
        stmt = self._build_select(criteria)
        stmt = self._apply_order(stmt, criteria.order_by)
        stmt = stmt.limit(criteria.limit).offset(criteria.offset)
        with session_scope(self._sm) as session:
            rows = session.scalars(stmt).all()
            return [RunRow.from_orm(r) for r in rows]

    def find_one(self, criteria: RunHistoryCriteria) -> Optional[RunRow]:
        results = self.find(replace(criteria, limit=1))
        return results[0] if results else None

    def count(self, criteria: RunHistoryCriteria) -> int:
        stmt = self._build_select(criteria, count=True)
        with session_scope(self._sm) as session:
            return int(session.scalar(stmt) or 0)

    def get_by_id(self, run_id: int) -> Optional[RunRow]:
        with session_scope(self._sm) as session:
            m = session.get(ActionLog, run_id)
            return RunRow.from_orm(m) if m else None

    def distinct_albums(self) -> List[str]:
        stmt = (
            select(ActionLog.source_album)
            .where(ActionLog.source_album.is_not(None))
            .where(ActionLog.source_album != "")
            .where(ActionLog.status != "reserved")
            .distinct()
            .order_by(ActionLog.source_album)
        )
        with session_scope(self._sm) as session:
            return [a for a in session.scalars(stmt).all() if a is not None]

    # -------------------------------------------------------------- mutations

    def start_action(
        self,
        action_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        self._check_writable()
        payload = payload or {}
        row = ActionLog(
            action_type=action_type,
            status="running",
            started_at=_now_iso(),
            payload_json=json.dumps(payload),
            **{k: payload.get(k) for k in ActionLog.hot_field_names()},
        )
        with session_scope(self._sm) as session:
            session.add(row)
            session.flush()
            return int(row.id)

    def complete_action(
        self,
        action_id: int,
        *,
        result_fields: Optional[Dict[str, Any]] = None,
        payload_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._check_writable()
        result_fields = result_fields or {}
        with session_scope(self._sm) as session:
            row = session.get(ActionLog, action_id)
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in action_log",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            ended = _now_iso()
            started = datetime.fromisoformat(row.started_at)
            row.status = "complete"
            row.ended_at = ended
            row.duration_s = (datetime.fromisoformat(ended) - started).total_seconds()
            for name in ActionLog.hot_field_names():
                new_val = result_fields.get(name)
                if new_val is not None:
                    setattr(row, name, new_val)
            payload = json.loads(row.payload_json or "{}")
            if payload_update:
                payload.update(payload_update)
            row.payload_json = json.dumps(payload)

    def fail_action(
        self,
        action_id: int,
        error_text: str,
        *,
        payload_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._check_writable()
        with session_scope(self._sm) as session:
            row = session.get(ActionLog, action_id)
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in action_log",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            ended = _now_iso()
            started = datetime.fromisoformat(row.started_at)
            row.status = "failed"
            row.ended_at = ended
            row.duration_s = (datetime.fromisoformat(ended) - started).total_seconds()
            row.error = error_text
            payload = json.loads(row.payload_json or "{}")
            if payload_update:
                payload.update(payload_update)
            row.payload_json = json.dumps(payload)

    def update_comment(self, action_id: int, comment: str) -> None:
        self._check_writable()
        if len(comment) > _COMMENT_MAX_LEN:
            raise ValidationError(
                f"Comment exceeds {_COMMENT_MAX_LEN} characters (got {len(comment)})",
                user_message=f"Comments are limited to {_COMMENT_MAX_LEN} characters.",
            )
        with session_scope(self._sm) as session:
            row = session.get(ActionLog, action_id)
            if row is None:
                raise NotFoundError(
                    f"action_id {action_id} not found in action_log",
                    user_message=f"Run #{action_id} doesn't exist.",
                )
            row.comment = comment

    # -------------------------------------------------------------- internals

    def _build_select(self, c: RunHistoryCriteria, count: bool = False):
        if c.date_from is not None and c.date_to is not None and c.date_from > c.date_to:
            raise ValidationError(
                f"date_from ({c.date_from}) > date_to ({c.date_to})",
                user_message="Invalid date range — 'from' is after 'to'.",
            )
        unknown = set(c.text_fields) - _TEXT_FIELD_ALLOWLIST
        if unknown:
            raise ValidationError(
                f"Unknown text_fields: {sorted(unknown)}",
                user_message="Internal error: invalid search field.",
            )

        if count:
            stmt = select(func.count()).select_from(ActionLog)
        else:
            stmt = select(ActionLog)

        stmt = stmt.where(ActionLog.status != "reserved")

        if c.ids:
            stmt = stmt.where(ActionLog.id.in_(c.ids))
        if c.album:
            stmt = stmt.where(ActionLog.source_album == c.album)
        if c.status:
            stmt = stmt.where(ActionLog.status == c.status)
        if c.producer:
            stmt = stmt.where(ActionLog.producer == c.producer)
        if c.parent_run_id is not None:
            stmt = stmt.where(ActionLog.parent_run_id == c.parent_run_id)
        if c.action_types:
            stmt = stmt.where(ActionLog.action_type.in_(c.action_types))
        if c.date_from is not None:
            stmt = stmt.where(ActionLog.started_at >= c.date_from.isoformat())
        if c.date_to is not None:
            stmt = stmt.where(
                ActionLog.started_at < func.date(c.date_to.isoformat(), "+1 day")
            )
        if c.text:
            like = f"%{c.text}%"
            field_to_col = {
                "source_album": ActionLog.source_album,
                "run_name": ActionLog.run_name,
                "comment": ActionLog.comment,
                "album": ActionLog.album,
                "run_id": ActionLog.run_id,
            }
            stmt = stmt.where(or_(*(field_to_col[f].like(like) for f in c.text_fields)))
        return stmt

    @staticmethod
    def _apply_order(stmt, order_by: str):
        if order_by == "started_at_asc":
            return stmt.order_by(ActionLog.started_at.asc())
        return stmt.order_by(ActionLog.started_at.desc())


__all__ = [
    "RunHistoryCriteria",
    "RunHistoryRepoConfig",
    "RunHistoryRepository",
]
