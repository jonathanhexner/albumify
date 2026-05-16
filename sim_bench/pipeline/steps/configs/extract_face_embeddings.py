"""Typed config for extract_face_embeddings step (spec-033 P-G)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractFaceEmbeddingsConfig(BaseModel):
    """Config for `extract_face_embeddings` step."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["insightface", "custom"] = Field(
        default="insightface",
        description=(
            "Embedding backend. 'insightface' uses buffalo_l's built-in "
            "w600k_r50; 'custom' uses the checkpoint at checkpoint_path."
        ),
    )
    checkpoint_path: str = Field(
        default="models/album_app/arcface_resnet50.pt",
        description="Path to custom ArcFace checkpoint (used when backend='custom').",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Inference device.",
    )
    model_name: str = Field(
        default="buffalo_l",
        description="InsightFace model pack (used when backend='insightface').",
    )
