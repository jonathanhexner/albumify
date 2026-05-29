"""Cluster ORM model — mirrors CLUSTERS_DDL."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class Cluster(Base):
    __tablename__ = "clusters"

    cluster_id: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    iteration: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    size: Mapped[int] = mapped_column(INTEGER, nullable=False)
    diameter: Mapped[Optional[float]] = mapped_column(REAL)
    avg_intra_dist: Mapped[Optional[float]] = mapped_column(REAL)
    origin: Mapped[str] = mapped_column(TEXT, nullable=False)
    parent_ids: Mapped[str] = mapped_column(TEXT, nullable=False)
