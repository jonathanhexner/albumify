"""Pipeline API schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class PipelineRequest(BaseModel):
    album_id: str
    steps: Optional[list[str]] = None
    config: Optional[dict[str, dict]] = None  # Runtime overrides (merged with profile)
    profile: Optional[str] = None  # Config profile name (uses default if not specified)
    fail_fast: bool = True
    runner: str = "local"  # spec-116: where it runs. Explicit: only an owner's choice ever starts the L4
    pipeline_name: Optional[str] = None  # the name the steps came from; default_pipeline if not given


class PipelineStatus(BaseModel):
    job_id: str
    album_id: str
    status: str
    current_step: Optional[str] = None
    progress: float = 0.0
    message: Optional[str] = None
    completed_steps: Optional[list] = None  # [{step, duration_ms, status, error}]
    total_steps: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    runner: Optional[str] = None  # spec-116 R7: None for runs made before it
    stage: Optional[str] = None   # queued | staging | uploading | running | copying back | loading

    class Config:
        from_attributes = True


class RunRow(BaseModel):
    """One row of the Albums page (spec-116 R10)."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    album_id: str
    album_name: str
    folder: str
    ran_on: str
    created_at: datetime
    photos: Optional[int] = None
    picks: Optional[int] = None
    people: Optional[int] = None
    status: str
    comparable: bool = False  # it has a finished job, so Compare runs can open it


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
    fc_export_dir: Optional[str] = None

    class Config:
        from_attributes = True
