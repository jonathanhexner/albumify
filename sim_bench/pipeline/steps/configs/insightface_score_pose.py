"""Typed config for insightface_score_pose step (spec-040 Phase 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InsightFaceScorePoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Inference device for the pose-from-5-point-landmarks scorer.",
    )
