"""Data models for the Streamlit frontend."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Step:
    """Pipeline step information."""
    name: str
    display_name: str
    description: str = ""
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    step_name: str
    success: bool
    duration_ms: float
    error_message: Optional[str] = None


@dataclass
class PipelineResult:
    """Full pipeline execution result."""
    success: bool
    total_duration_ms: float
    step_results: List[StepResult] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ImageInfo:
    """Information about a single image."""
    path: str
    filename: str
    iqa_score: Optional[float] = None
    ava_score: Optional[float] = None
    composite_score: Optional[float] = None
    face_count: int = 0
    cluster_id: Optional[int] = None
    is_selected: bool = False
    sharpness: Optional[float] = None
    face_pose_scores: Optional[List[float]] = None
    face_eyes_scores: Optional[List[float]] = None
    face_smile_scores: Optional[List[float]] = None
    thumbnail_url: Optional[str] = None
    # InsightFace person detection metrics
    person_detected: Optional[bool] = None
    body_facing_score: Optional[float] = None
    person_confidence: Optional[float] = None


@dataclass
class ClusterInfo:
    """Information about an image cluster."""
    cluster_id: int
    image_count: int
    selected_count: int
    has_faces: bool
    face_count: Optional[int] = None
    images: List[ImageInfo] = field(default_factory=list)
    person_labels: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Face:
    """Detected face information."""
    face_id: str
    image_path: str
    bbox: tuple  # (x, y, w, h)
    confidence: float
    identity_id: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class Person:
    """Person (identity cluster) information."""
    person_id: str
    name: Optional[str] = None
    face_count: int = 0
    image_count: int = 0
    representative_face: Optional[str] = None  # Path to representative image
    thumbnail_bbox: Optional[List[float]] = None  # [x, y, w, h] for face crop
    images: List[str] = field(default_factory=list)


@dataclass
class Album:
    """Album information."""
    album_id: str
    name: str
    source_directory: str
    created_at: datetime = field(default_factory=datetime.now)
    total_images: int = 0
    selected_images: int = 0
    status: PipelineStatus = PipelineStatus.IDLE
    clusters: List[ClusterInfo] = field(default_factory=list)
    people: List[Person] = field(default_factory=list)


@dataclass
class PipelineProgress:
    """Current pipeline progress."""
    status: PipelineStatus
    current_step: Optional[str] = None
    current_step_progress: float = 0.0
    current_step_message: str = ""
    completed_steps: List[str] = field(default_factory=list)
    total_steps: int = 0


@dataclass
class ApiResponse:
    """Generic API response wrapper."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ExportOptions:
    """Export configuration options."""
    output_directory: str
    copy_files: bool = True
    create_structure: bool = True  # Maintain cluster structure
    include_metadata: bool = True
    format: str = "folder"  # folder, zip
