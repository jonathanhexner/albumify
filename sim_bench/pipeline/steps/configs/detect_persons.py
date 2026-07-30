"""Typed config for detect_persons step (spec-040 Phase 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectPersonsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_size: Literal["nano", "small", "medium"] = Field(
        default="small",
        description="YOLOv8-Pose model size.",
    )
    confidence_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Min detection confidence for a person bbox.",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Inference device.",
    )
    keypoint_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Min per-keypoint confidence to count a keypoint as visible.",
    )
    orientation_strategy: Literal["shoulder_hip", "torso", "head"] = Field(
        default="shoulder_hip",
        description="Body-orientation heuristic.",
    )
