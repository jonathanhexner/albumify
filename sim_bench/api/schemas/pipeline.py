"""Pipeline API schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class PipelineRequest(BaseModel):
    album_id: str
    steps: Optional[list[str]] = None
    config: Optional[dict[str, dict]] = None  # Runtime overrides (merged with profile)
    profile: Optional[str] = None  # Config profile name (uses default if not specified)
    fail_fast: bool = True


class PipelineStatus(BaseModel):
    job_id: str
    album_id: str
    status: str
    current_step: Optional[str] = None
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PipelineResultResponse(BaseModel):
    job_id: str
    album_id: str
    total_images: int
    filtered_images: int
    num_clusters: int
    num_selected: int
    scene_clusters: dict[int, list[str]]
    selected_images: list[str]
    step_timings: dict[str, int]
    total_duration_ms: int

    class Config:
        from_attributes = True
