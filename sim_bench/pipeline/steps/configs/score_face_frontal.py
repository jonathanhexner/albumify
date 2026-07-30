"""Typed config for score_face_frontal step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScoreFaceFrontalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_frontal_score: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Faces below this frontal score are marked non-clusterable.",
    )
    min_eye_bbox_ratio: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Inter-eye / bbox width threshold (below = profile).",
    )
    max_asymmetry: float = Field(
        default=1.8, ge=0.0,
        description="Nose-eye asymmetry threshold (above = profile).",
    )
