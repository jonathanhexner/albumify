"""Typed config for filter_quality step (spec-033 P-G)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FilterQualityConfig(BaseModel):
    """Config for `filter_quality` step.

    Image-level technical-quality gate. Rejects images with low IQA or sharpness
    before they reach face detection / clustering.
    """

    model_config = ConfigDict(extra="forbid")

    min_iqa_score: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description=(
            "Dimensionless 0-1. Minimum NIMA-IQA technical-quality score. "
            "Filter: image_quality."
        ),
    )
    min_sharpness: float = Field(
        default=0.2,
        ge=0.0, le=1.0,
        description=(
            "Dimensionless 0-1. Minimum Laplacian-variance sharpness. "
            "Filter: image_quality."
        ),
    )
