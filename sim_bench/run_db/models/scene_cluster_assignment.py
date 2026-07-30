"""SceneClusterAssignment ORM model — mirrors SCENE_CLUSTER_ASSIGNMENTS_DDL."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class SceneClusterAssignment(Base):
    __tablename__ = "scene_cluster_assignments"

    image_path: Mapped[str] = mapped_column(
        TEXT, ForeignKey("images.image_path"), primary_key=True, nullable=False
    )
    scene_cluster_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    iteration: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    distance_to_centroid: Mapped[Optional[float]] = mapped_column(REAL)

    __table_args__ = (
        Index("idx_sca_cluster", "scene_cluster_id", "iteration"),
    )
