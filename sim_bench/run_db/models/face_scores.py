"""FaceScores ORM model — mirrors FACE_SCORES_DDL."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import INTEGER, REAL, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from sim_bench.run_db.models._base import Base


class FaceScores(Base):
    __tablename__ = "face_scores"

    face_id: Mapped[int] = mapped_column(
        INTEGER, ForeignKey("faces.face_id"), primary_key=True
    )
    pose_score: Mapped[Optional[float]] = mapped_column(REAL)
    eyes_score: Mapped[Optional[float]] = mapped_column(REAL)
    expression_score: Mapped[Optional[float]] = mapped_column(REAL)
    frontal_score: Mapped[Optional[float]] = mapped_column(REAL)
    is_clusterable: Mapped[Optional[int]] = mapped_column(INTEGER)
