"""Typed config for build_scene_distance step (spec-103)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BuildSceneDistanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boost: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Strength of the short-range time pull (0=off, 1=collapse near-simultaneous photos).",
    )
    tau_sec: float = Field(
        default=60.0, gt=0.0,
        description="Time constant in seconds; photos within ~this gap get the boost, beyond ~3x negligible.",
    )
