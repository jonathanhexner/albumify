"""ClusterAssignment ORM model — mirrors CLUSTER_ASSIGNMENTS_DDL."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"

    face_id: Mapped[int] = mapped_column(
        INTEGER, ForeignKey("faces.face_id"), primary_key=True, nullable=False
    )
    cluster_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    iteration: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    is_exemplar: Mapped[int] = mapped_column(
        INTEGER, nullable=False, server_default=text("0")
    )
    d10_score: Mapped[Optional[float]] = mapped_column(REAL)

    __table_args__ = (
        Index("idx_assign_iter", "iteration"),
        Index("idx_assign_cluster", "cluster_id", "iteration"),
    )
