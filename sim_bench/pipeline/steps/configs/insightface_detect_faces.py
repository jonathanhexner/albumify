"""Typed config for insightface_detect_faces step (spec-033 P-G)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InsightFaceDetectFacesConfig(BaseModel):
    """Config for `insightface_detect_faces` step (SCRFD backbone)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="buffalo_l",
        description=(
            "InsightFace model pack name. 'buffalo_l' is the default "
            "(SCRFD 10G detection + ArcFace w600k_r50 embedding)."
        ),
    )
    detection_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description=(
            "Dimensionless 0-1. SCRFD detection score threshold. "
            "Higher = stricter detection."
        ),
    )
    min_face_size: int = Field(
        default=50,
        ge=0,
        description=(
            "Pixels. Minimum face bbox edge to keep. Note: this is the "
            "absolute-px filter — the relative face-size rejector is "
            "filter_faces.min_bbox_ratio."
        ),
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Inference device for SCRFD.",
    )
    associate_to_person: bool = Field(
        default=True,
        description=(
            "If True, associate each detected face to a person from "
            "detect_persons by spatial overlap."
        ),
    )
