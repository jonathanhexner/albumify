"""ActionLog ORM model — the action_log table.

Replaces the _COLUMNS / ColumnDef registry from spec-044. Columns and
types match the production schema exactly. Schema evolution happens
through Alembic migrations, not by editing this file then hand-writing
ALTER TABLE.

Column metadata (spec-048):
- ``info={"updated_on_complete": True}`` marks fields the Repository
  copies from a caller-supplied result-dict when a run is completed.
  Read back via :py:meth:`ActionLog.hot_field_names`.

Indexes match what the legacy _create_table_sql() + _migrate() produced:
- (started_at DESC)
- (action_type, started_at DESC)
- (source_album)
- (comment)
"""
from __future__ import annotations

from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from face_cluster.repositories._orm_base import Base

# Sentinel placed on every column the Repository allows callers to set
# (or update) via the result-fields argument of ``complete_action``.
# Centralising this on the column means there's no parallel tuple to
# keep in sync — see spec-048 smell S3.
_HOT = {"updated_on_complete": True}


class ActionLog(Base):
    __tablename__ = "action_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="running")
    started_at: Mapped[str] = mapped_column(Text, nullable=False)
    ended_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run_id: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    source_dir: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    output_dir: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    album: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    n_faces: Mapped[int | None] = mapped_column(Integer, nullable=True, info=_HOT)
    n_clusters: Mapped[int | None] = mapped_column(Integer, nullable=True, info=_HOT)
    n_noise: Mapped[int | None] = mapped_column(Integer, nullable=True, info=_HOT)
    log_file: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)

    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_album: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    run_name: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    parent_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, info=_HOT)
    run_kind: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)
    n_core: Mapped[int | None] = mapped_column(Integer, nullable=True, info=_HOT)

    producer: Mapped[str | None] = mapped_column(Text, nullable=True, info=_HOT)

    __table_args__ = (
        Index("idx_action_log_started", "started_at"),
        Index("idx_action_log_type", "action_type", "started_at"),
        Index("idx_action_log_source_album", "source_album"),
        Index("idx_action_log_comment", "comment"),
    )

    @classmethod
    def hot_field_names(cls) -> tuple[str, ...]:
        """Columns the Repository copies from ``result_fields`` on completion.

        Read directly from per-column ``info`` metadata. The order is the
        column-declaration order so callers (and the equivalence test)
        see a stable sequence.
        """
        return tuple(
            c.name for c in cls.__table__.columns
            if c.info.get("updated_on_complete")
        )


__all__ = ["ActionLog"]
