"""Typed config for align_faces step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AlignFacesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_size: int = Field(
        default=256, ge=32,
        description="Output aligned face size (pixels per side).",
    )
    skip_filtered: bool = Field(
        default=True,
        description="Skip faces with filter_passed=False to save compute.",
    )
