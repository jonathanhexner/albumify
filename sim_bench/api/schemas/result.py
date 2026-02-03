"""Pydantic schemas for Results API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ImageMetrics(BaseModel):
    """Metrics for a single image."""
    path: str
    iqa_score: Optional[float] = None
    ava_score: Optional[float] = None
    composite_score: Optional[float] = None
    sharpness: Optional[float] = None
    cluster_id: Optional[int] = None
    face_count: Optional[int] = None
    face_pose_scores: Optional[list[float]] = None
    face_eyes_scores: Optional[list[float]] = None
    face_smile_scores: Optional[list[float]] = None
    is_selected: bool = False


class ClusterInfo(BaseModel):
    """Information about a cluster."""
    cluster_id: int
    image_count: int
    selected_count: int = 0
    has_faces: bool = False
    face_count: int = 0
    images: list[ImageMetrics] = []
    best_image: Optional[str] = None
    person_labels: dict[str, list[str]] = {}


class PipelineMetrics(BaseModel):
    """Aggregate metrics for a pipeline run."""
    total_images: int
    filtered_images: int
    num_clusters: int
    num_selected: int
    num_people: Optional[int] = None
    avg_iqa_score: Optional[float] = None
    avg_ava_score: Optional[float] = None
    step_timings: dict[str, int]
    total_duration_ms: int


class ResultSummary(BaseModel):
    """Summary of pipeline results."""
    job_id: str
    album_id: str
    album_name: str
    status: str
    total_images: int
    filtered_images: int
    num_clusters: int
    num_selected: int
    num_people: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[int] = None

    class Config:
        from_attributes = True


class ResultDetail(BaseModel):
    """Full pipeline result details."""
    job_id: str
    album_id: str
    album_name: str
    status: str
    pipeline_name: str
    steps: list[str]

    # Counts
    total_images: int
    filtered_images: int
    num_clusters: int
    num_selected: int
    num_people: Optional[int] = None

    # Results
    scene_clusters: dict[int, list[str]]
    selected_images: list[str]

    # Timing
    step_timings: dict[str, int]
    total_duration_ms: int

    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ExportRequest(BaseModel):
    """Request to export results."""
    output_path: str
    include_selected: bool = True
    include_all_filtered: bool = False
    organize_by_cluster: bool = False
    organize_by_person: bool = False
    copy_mode: str = "copy"  # "copy" or "symlink"


class ExportResponse(BaseModel):
    """Response from export operation."""
    success: bool
    output_path: str
    files_exported: int
    errors: list[str] = []
