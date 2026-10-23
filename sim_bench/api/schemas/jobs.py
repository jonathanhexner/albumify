"""Response / request models for the jobs API (spec-112 TA5.1). Mirrors sim_bench/album/schema.py."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobSubmit(_Strict):
    source: str = Field(..., description="Local folder of photos (Phase A: local runs only)")
    runner: Optional[str] = Field(None, description="local | modal_gpu | modal_cpu; default from configs/serving.yaml")
    pipeline: Optional[str] = None
    job_id: Optional[str] = None
    start: int = 0
    n: Optional[int] = Field(None, ge=1)


class JobLoad(_Strict):
    runner: str = "modal_gpu"
    source: Optional[str] = None


class JobAccepted(_Strict):
    job_id: str
    status: str


class JobEventOut(_Strict):
    ts: datetime
    kind: str
    message: str


class JobOut(_Strict):
    id: str
    status: str
    runner: str
    pipeline: str
    source: Optional[str] = None
    tag: Optional[str] = None
    n_inputs: int
    error: Optional[str] = None
    failed_step: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class JobDetail(JobOut):
    events: List[JobEventOut] = []


class ScoreOut(_Strict):
    composite: Optional[float] = None
    sharpness: Optional[float] = None
    aesthetic: Optional[float] = None
    occlusion: Optional[float] = None
    tilt: Optional[float] = None


class PickOut(_Strict):
    id: str
    order: int
    score: ScoreOut
    scene_cluster: int
    beat: List[str]
    people: List[str]
    caption: str


class DroppedOut(_Strict):
    id: str
    reason: str
    reason_detail: str
    beaten_by: List[str]
    promotable: bool


class PersonOut(_Strict):
    id: str
    face_count: int
    sample_ids: List[str]
    label: str


class CountsOut(_Strict):
    uploaded: int
    kept: int
    people: int
    scenes: int
    set_aside: int


class AlbumOut(_Strict):
    job_id: str
    created_at: str
    counts: CountsOut
    picks: List[PickOut]
    dropped: List[DroppedOut]
    people: List[PersonOut]
    theme: Dict[str, Any]


class PhotoOut(_Strict):
    photo_id: str
    in_album: bool
    status: Optional[str] = None
    reason_detail: str
    taken_at: Optional[str] = None
    scene_id: Optional[int] = None
    composite: Optional[float] = None
    has_display: bool
    n_faces: int
    decisions: List[Dict[str, Any]]


class FaceOut(_Strict):
    face_id: int
    photo_id: str
    person_id: Optional[int] = None
    det_score: Optional[float] = None
    is_core: bool
    rejection_reason: Optional[str] = None
    has_crop: bool


class EditIn(_Strict):
    kind: str = Field(..., description="promote | demote | rename")
    payload: Dict[str, Any]


class EditOut(_Strict):
    id: int
    kind: str
    payload: Dict[str, Any]


class CheckOut(_Strict):
    name: str
    baseline: int
    current: int
    allowed: int
    ok: bool


class PickDiffOut(_Strict):
    same: List[str]
    swapped: List[List[str]]          # [A pick, B pick from the same near-duplicate group]
    missing: List[str]
    extra: List[str]
    also_kept: List[str]


class PhotoDecisionOut(_Strict):
    state: str                        # pick | near_dup | not_selected | set_aside | removed
    detail: str
    scene: Optional[int] = None
    score: Optional[float] = None


class ComparePersonOut(_Strict):
    id: str
    label: str
    face_count: int
    face_id: Optional[int] = None     # one face crop to show


class CompareOut(_Strict):
    """spec-114: job ``b`` (new) against job ``a`` (reference) on the same photos."""
    a: str
    b: str
    ok: bool
    failures: List[str]
    metrics: Dict[str, float]
    checks: List[CheckOut]
    picks: PickDiffOut
    decisions: Dict[str, Dict[str, PhotoDecisionOut]]
    people: Dict[str, List[ComparePersonOut]]
