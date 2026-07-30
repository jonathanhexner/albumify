"""Image ORM model — mirrors IMAGES_DDL (spec-040 Phase 4, schema v5)."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT, Index, text
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class Image(Base):
    __tablename__ = "images"

    image_path: Mapped[str] = mapped_column(TEXT, primary_key=True)
    image_id: Mapped[Optional[str]] = mapped_column(TEXT)
    width_px: Mapped[Optional[int]] = mapped_column(INTEGER)
    height_px: Mapped[Optional[int]] = mapped_column(INTEGER)
    n_faces: Mapped[int] = mapped_column(
        INTEGER, nullable=False, server_default=text("0")
    )
    iqa_score: Mapped[Optional[float]] = mapped_column(REAL)
    ava_score: Mapped[Optional[float]] = mapped_column(REAL)
    sharpness_score: Mapped[Optional[float]] = mapped_column(REAL)
    composite_score: Mapped[Optional[float]] = mapped_column(REAL)
    scene_cluster_id: Mapped[Optional[int]] = mapped_column(INTEGER)
    filter_passed: Mapped[int] = mapped_column(
        INTEGER, nullable=False, server_default=text("1")
    )
    created_at: Mapped[str] = mapped_column(TEXT, nullable=False)

    __table_args__ = (
        Index("idx_images_scene", "scene_cluster_id"),
    )
