"""spec-033 P-D: typed "click an image, see everything" return type.

Extends spec-023 (Image Detail View) by routing the data through
``RunStore.image_detail`` instead of recomputing from JSON manifests at
read time. One SQL JOIN replaces the existing scatter of context-dict
lookups + cache reads.

The model lives outside ``run_store.py`` so the FC App popup and the
Albumify popup can both depend on it without circular imports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class FaceFilterDecision(BaseModel):
    """One filter's verdict on one face (the per-face slice of spec-032)."""

    model_config = ConfigDict(extra="forbid")

    filter_name: str = Field(description="Canonical filter name from KNOWN_FILTERS.")
    rejected:    bool = Field(description="True if the filter rejected this face.")
    reason:      str  = Field(description="Human-readable summary of the verdict.")
    measured:    Dict = Field(default_factory=dict, description="Measured values that produced the verdict.")


class FaceDetail(BaseModel):
    """Everything we know about one face from one run."""

    model_config = ConfigDict(extra="forbid")

    face_id:    int = Field(description="Primary key in the faces table.")
    face_index: Optional[int] = Field(default=None, description="Index of this face within its source image.")
    bbox:       Tuple[float, float, float, float] = Field(description="(x, y, w, h) — units match what the producer wrote.")
    crop_path:  Optional[str] = Field(default=None, description="Path to the aligned 112×112 crop relative to run dir.")
    det_score:  Optional[float] = Field(default=None, description="Detection confidence (0-1) if available.")
    blur_score: float = Field(description="Laplacian variance blur score (>=0).")
    area:       float = Field(description="Face area — UNIT varies by producer (SIGHTING-060).")
    pose:       Optional[Tuple[float, float, float]] = Field(default=None, description="(yaw, pitch, roll) in degrees.")
    is_core:    bool = Field(description="Whether the face passed quality gating.")
    rejection_reason: Optional[str] = Field(default=None, description="First failing gate name, if any.")
    cluster_id: Optional[int] = Field(default=None, description="Final cluster id (post-merge) or None for noise.")
    is_exemplar: bool = Field(default=False, description="True if this face is an exemplar of its cluster.")
    filter_decisions: List[FaceFilterDecision] = Field(
        default_factory=list,
        description="spec-032 typed filter log for this face.",
    )


class ImageDetail(BaseModel):
    """Everything we know about one image from one run.

    Returned by ``RunStore.image_detail(image_path)`` — the canonical answer to
    the user's "click an image, see everything we know about it" ask.
    """

    model_config = ConfigDict(extra="forbid")

    image_path:       str = Field(description="The path queried.")
    image_id:         Optional[str] = Field(default=None, description="Image id used internally (often the basename).")
    iqa_score:        Optional[float] = Field(default=None, description="filter_quality IQA score (0-1).")
    ava_score:        Optional[float] = Field(default=None, description="select_best aesthetics score (0-1).")
    sharpness_score:  Optional[float] = Field(default=None, description="Laplacian-variance sharpness (0-1).")
    scene_cluster_id: Optional[int] = Field(default=None, description="Scene cluster id from cluster_scenes.")
    faces:            List[FaceDetail] = Field(
        default_factory=list,
        description="Every face detected on this image in this run.",
    )
    image_filter_decisions: List[FaceFilterDecision] = Field(
        default_factory=list,
        description="spec-032 image-level filter decisions (e.g. image_quality).",
    )
