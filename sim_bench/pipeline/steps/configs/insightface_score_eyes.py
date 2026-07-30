"""Typed config for insightface_score_eyes step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InsightFaceScoreEyesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_face_size: int = Field(default=50, ge=0, description="Skip faces below this px size.")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Min detection confidence to score.")
    crop_margin: float = Field(default=0.3, ge=0.0, le=1.0, description="Padding around bbox crop (0.2–0.4).")
    target_size: int = Field(default=256, ge=32, description="Resize aligned crop to NxN pixels before scoring.")
    ear_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Eye-aspect-ratio threshold for open/closed.",
    )
