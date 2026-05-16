"""Typed config for filter_faces step (spec-033 P-G)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FilterFacesConfig(BaseModel):
    """Config for `filter_faces` step.

    Face-level filter that rejects small or low-confidence faces before
    they reach scoring / clustering. Runs after `insightface_detect_faces`.
    """

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description=(
            "Dimensionless 0-1. Minimum InsightFace detection confidence. "
            "Filter: face_confidence."
        ),
    )
    min_bbox_ratio: float = Field(
        default=0.02,
        ge=0.0, le=1.0,
        description=(
            "bbox_width / image_width. Minimum face size relative to image. "
            "Filter: face_bbox_ratio. (spec-033 P-A surfaced from hidden default.)"
        ),
    )
    min_relative_size: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description=(
            "bbox_width / max_bbox_width_in_image. Rejects faces much smaller "
            "than the largest face on the same image. Filter: face_relative_size."
        ),
    )
    min_eye_ratio: float = Field(
        default=0.01,
        ge=0.0, le=1.0,
        description=(
            "inter_eye_distance / image_width. Minimum inter-eye distance "
            "relative to image. Filter: face_eye_ratio."
        ),
    )
