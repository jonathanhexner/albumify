"""Pydantic schemas for face management API."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class FaceInfo(BaseModel):
    """Complete information about a single face."""

    # Identity
    face_key: str = Field(..., description="Unique face identifier: image_path:face_N")
    image_path: str = Field(..., description="Path to source image")
    face_index: int = Field(..., description="Face index within the image")

    # Visual
    thumbnail_base64: Optional[str] = Field(None, description="Base64 encoded JPEG thumbnail")
    bbox: Optional[dict] = Field(None, description="Bounding box {x, y, w, h} in relative coords")

    # Classification
    status: str = Field(..., description="assigned | unassigned | untagged | not_a_face")
    person_id: Optional[str] = Field(None, description="Assigned person ID")
    person_name: Optional[str] = Field(None, description="Assigned person name")

    # Assignment details
    assignment_method: Optional[str] = Field(None, description="core | auto | user")
    assignment_confidence: Optional[float] = Field(None, description="Confidence of assignment")

    # Quality metrics (for debug view)
    frontal_score: Optional[float] = Field(None, description="Face frontal score 0-1")
    centroid_distance: Optional[float] = Field(None, description="Distance to cluster centroid")
    exemplar_matches: Optional[int] = Field(None, description="Number of matching exemplars")

    class Config:
        from_attributes = True


class PersonDistance(BaseModel):
    """Distance from a face to a specific person."""

    person_id: str
    person_name: str
    thumbnail_base64: Optional[str] = Field(None, description="Person representative thumbnail")

    centroid_distance: float = Field(..., description="Distance to person's centroid")
    exemplar_matches: int = Field(..., description="Number of exemplars within threshold")
    min_exemplar_distance: float = Field(..., description="Distance to closest exemplar")
    would_attach: bool = Field(..., description="Would meet attachment criteria if assigned")


class BorderlineFace(BaseModel):
    """A face in the uncertainty zone needing user decision."""

    face: FaceInfo

    # Closest match
    closest_person_id: str
    closest_person_name: str
    closest_person_thumbnail: Optional[str] = None
    distance: float

    # Uncertainty (0 = very uncertain, 1 = clear decision)
    uncertainty_score: float

    # Thresholds for context
    attach_threshold: float
    reject_threshold: float


class PersonSummary(BaseModel):
    """Summary of a person for listing."""

    person_id: str
    name: str
    thumbnail_base64: Optional[str] = None
    face_count: int
    exemplar_count: int = 0
    cluster_tightness: Optional[float] = Field(None, description="Average internal distance")


class FaceAction(BaseModel):
    """Single face action in a batch."""

    face_key: str = Field(..., description="Face to act on")
    action: str = Field(..., description="assign | unassign | untag | not_a_face")
    target_person_id: Optional[str] = Field(None, description="Target person for assign/reassign")
    new_person_name: Optional[str] = Field(None, description="Name for creating new person")


class BatchChangeRequest(BaseModel):
    """Request to apply multiple face changes."""

    changes: List[FaceAction]
    recluster: bool = Field(True, description="Trigger reclustering after apply")


class BatchChangeResponse(BaseModel):
    """Response after applying batch changes."""

    applied_count: int
    failed_count: int
    failures: List[dict] = Field(default_factory=list, description="List of {face_key, error}")

    # Reclustering results (if recluster=True)
    auto_assigned_count: int = 0
    new_unassigned_count: int = 0


class FaceOverrideResponse(BaseModel):
    """Response for a face override record."""

    id: str
    album_id: str
    run_id: Optional[str]
    face_key: str
    status: str
    person_id: Optional[str]
    created_at: datetime
    created_by: str

    class Config:
        from_attributes = True
