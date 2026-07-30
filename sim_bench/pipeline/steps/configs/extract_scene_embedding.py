"""Typed config for extract_scene_embedding step (spec-040 Phase 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractSceneEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["dinov2", "openclip", "resnet50"] = Field(
        default="dinov2",
        description="Scene-embedding backbone model.",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Inference device.",
    )
