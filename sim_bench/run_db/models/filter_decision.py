"""FilterDecision ORM model — mirrors FILTER_DECISIONS_DDL (spec-032)."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, TEXT, Index
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class FilterDecision(Base):
    __tablename__ = "filter_decisions"

    item_id: Mapped[str] = mapped_column(TEXT, primary_key=True, nullable=False)
    item_type: Mapped[str] = mapped_column(TEXT, nullable=False)
    parent_id: Mapped[Optional[str]] = mapped_column(TEXT)
    filter_name: Mapped[str] = mapped_column(TEXT, primary_key=True, nullable=False)
    rejected: Mapped[int] = mapped_column(INTEGER, nullable=False)
    reason: Mapped[str] = mapped_column(TEXT, nullable=False)
    measured_json: Mapped[str] = mapped_column(TEXT, nullable=False)

    __table_args__ = (
        Index("idx_fd_filter", "filter_name"),
        Index("idx_fd_item", "item_type", "item_id"),
    )
