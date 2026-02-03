"""Pydantic schemas for People API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class FaceInstance(BaseModel):
    """A single face instance belonging to a person."""
    image_path: str
    face_index: int
    bbox: Optional[list[float]] = None  # [x, y, width, height]
    score: Optional[float] = None  # Quality score


class PersonResponse(BaseModel):
    """Response model for a detected person."""
    id: str
    album_id: str
    run_id: str
    person_index: int
    name: Optional[str] = None
    thumbnail_image_path: Optional[str] = None
    thumbnail_face_index: Optional[int] = None
    thumbnail_bbox: Optional[list[float]] = None
    face_count: int
    image_count: int
    face_instances: list[FaceInstance] = []
    created_at: datetime

    class Config:
        from_attributes = True


class PersonListResponse(BaseModel):
    """Summary response for listing people."""
    id: str
    person_index: int
    name: Optional[str] = None
    thumbnail_image_path: Optional[str] = None
    thumbnail_bbox: Optional[list[float]] = None
    face_count: int
    image_count: int

    class Config:
        from_attributes = True


class PersonRenameRequest(BaseModel):
    """Request to rename a person."""
    name: str


class PersonMergeRequest(BaseModel):
    """Request to merge multiple people into one."""
    person_ids: list[str]


class PersonSplitRequest(BaseModel):
    """Request to split a person into multiple people."""
    face_indices: list[int]  # Indices of faces to split out


class PersonImageResponse(BaseModel):
    """Response for a person's image."""
    image_path: str
    face_count: int  # Number of faces of this person in the image
    faces: list[FaceInstance]
