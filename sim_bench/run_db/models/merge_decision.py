"""MergeDecision ORM model — mirrors MERGE_DECISIONS_DDL.

28-column layout (matches MergeDecisionRow.field_names() — spec-030 FR-004).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT, Index
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class MergeDecision(Base):
    __tablename__ = "merge_decisions"

    iteration: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    cluster_a: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    cluster_b: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    cluster_a_size: Mapped[int] = mapped_column(INTEGER, nullable=False)
    cluster_b_size: Mapped[int] = mapped_column(INTEGER, nullable=False)
    exemplar_dist: Mapped[float] = mapped_column(REAL, nullable=False)
    threshold_used: Mapped[float] = mapped_column(REAL, nullable=False)
    T_a: Mapped[Optional[float]] = mapped_column(REAL)
    T_b: Mapped[Optional[float]] = mapped_column(REAL)
    T_global: Mapped[Optional[float]] = mapped_column(REAL)
    p25_cross_dist: Mapped[Optional[float]] = mapped_column(REAL)
    passes_cross: Mapped[Optional[int]] = mapped_column(INTEGER)
    support: Mapped[int] = mapped_column(INTEGER, nullable=False)
    unique_support: Mapped[Optional[int]] = mapped_column(INTEGER)
    required_support: Mapped[int] = mapped_column(INTEGER, nullable=False)
    post_diameter: Mapped[float] = mapped_column(REAL, nullable=False)
    max_allowed_diameter: Mapped[float] = mapped_column(REAL, nullable=False)
    margin_gap: Mapped[float] = mapped_column(REAL, nullable=False)
    margin_dist_to_b: Mapped[float] = mapped_column(REAL, nullable=False)
    margin_competitor_dist: Mapped[float] = mapped_column(REAL, nullable=False)
    margin_competitor_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    passes_exemplar: Mapped[int] = mapped_column(INTEGER, nullable=False)
    passes_support: Mapped[int] = mapped_column(INTEGER, nullable=False)
    passes_margin: Mapped[int] = mapped_column(INTEGER, nullable=False)
    passes_diameter: Mapped[int] = mapped_column(INTEGER, nullable=False)
    action: Mapped[str] = mapped_column(TEXT, nullable=False)
    actually_merged: Mapped[int] = mapped_column(INTEGER, nullable=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(TEXT)

    __table_args__ = (
        Index("idx_md_iter", "iteration"),
        Index("idx_md_pair", "cluster_a", "cluster_b"),
    )
