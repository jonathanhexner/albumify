"""Pydantic schemas for Results API."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class ImageMetrics(BaseModel):
    """Metrics for a single image.

    spec-085: this is the SINGLE SOURCE OF TRUTH for the per-image API contract.
    It must declare every field ``_build_image_metrics`` emits — a parity test
    enforces that. FastAPI's ``response_model`` drops any undeclared field, so a
    field missing here silently never reaches the UI (the spec-084 blank-columns
    bug). Add new per-image metrics HERE; ``_build_image_dict`` derives from it.
    """
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
    # spec-084 composite breakdown + decision reason.
    quality_score: Optional[float] = None
    person_penalty: Optional[float] = None
    filter_reason: Optional[str] = None
    # spec-085: person detection + frontal scoring + filter stats (were dropped).
    person_detected: Optional[bool] = None
    body_facing_score: Optional[float] = None
    person_confidence: Optional[float] = None
    best_frontal_score: Optional[float] = None
    best_centrality: Optional[float] = None
    roll_angles: Optional[list[float]] = None
    filter_stats: Optional[dict[str, Any]] = None
    filter_scores: Optional[list[dict[str, Any]]] = None
    frontal_stats: Optional[dict[str, Any]] = None
    frontal_scores: Optional[list[dict[str, Any]]] = None


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
    fc_export_dir: Optional[str] = None
    step_decisions: Optional[list] = None
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

    fc_export_dir: Optional[str] = None

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
