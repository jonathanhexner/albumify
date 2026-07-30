"""Typed config for detect_face_orientation step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DetectFaceOrientationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_orientations: bool = Field(
        default=True,
        description="Log non-zero face orientations for debugging.",
    )
