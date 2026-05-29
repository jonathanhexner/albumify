"""Face ORM model — mirrors FACES_DDL.

28 columns at schema v5 (added *_ratio columns + scene_cluster_id).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, TEXT
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class Face(Base):
    __tablename__ = "faces"

    face_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    image_path: Mapped[Optional[str]] = mapped_column(TEXT)
    image_id: Mapped[Optional[str]] = mapped_column(TEXT)
    face_index: Mapped[Optional[int]] = mapped_column(INTEGER)
    bbox_x: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_y: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_w: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_h: Mapped[Optional[float]] = mapped_column(REAL)
    crop_path: Mapped[Optional[str]] = mapped_column(TEXT)
    det_score: Mapped[Optional[float]] = mapped_column(REAL)
    blur_score: Mapped[Optional[float]] = mapped_column(REAL)
    area: Mapped[Optional[float]] = mapped_column(REAL)
    yaw: Mapped[Optional[float]] = mapped_column(REAL)
    pitch: Mapped[Optional[float]] = mapped_column(REAL)
    roll: Mapped[Optional[float]] = mapped_column(REAL)
    is_core: Mapped[int] = mapped_column(INTEGER, nullable=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(TEXT)
    iqa_score: Mapped[Optional[float]] = mapped_column(REAL)
    ava_score: Mapped[Optional[float]] = mapped_column(REAL)
    sharpness_score: Mapped[Optional[float]] = mapped_column(REAL)
    scene_cluster_id: Mapped[Optional[int]] = mapped_column(INTEGER)
    area_ratio: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_x_ratio: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_y_ratio: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_w_ratio: Mapped[Optional[float]] = mapped_column(REAL)
    bbox_h_ratio: Mapped[Optional[float]] = mapped_column(REAL)
