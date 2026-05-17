"""Typed config for cluster_scenes step (spec-040 Phase 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClusterScenesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["hdbscan", "dbscan", "kmeans"] = Field(
        default="hdbscan", description="Scene clustering algorithm.",
    )
    min_cluster_size: int = Field(
        default=2, ge=1,
        description="Minimum images per scene cluster.",
    )
    min_samples: int = Field(
        default=2, ge=1, description="HDBSCAN min_samples parameter.",
    )
