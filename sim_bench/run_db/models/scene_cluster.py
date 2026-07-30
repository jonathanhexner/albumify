"""SceneCluster ORM model — mirrors SCENE_CLUSTERS_DDL (spec-040 Phase 4)."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class SceneCluster(Base):
    __tablename__ = "scene_clusters"

    scene_cluster_id: Mapped[int] = mapped_column(
        INTEGER, primary_key=True, nullable=False
    )
    iteration: Mapped[int] = mapped_column(INTEGER, primary_key=True, nullable=False)
    size: Mapped[int] = mapped_column(INTEGER, nullable=False)
    method: Mapped[str] = mapped_column(TEXT, nullable=False)
    exemplar_image_path: Mapped[Optional[str]] = mapped_column(TEXT)
    avg_intra_distance: Mapped[Optional[float]] = mapped_column(REAL)
    created_at: Mapped[str] = mapped_column(TEXT, nullable=False)
