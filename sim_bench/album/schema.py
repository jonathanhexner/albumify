"""Normalized album result (spec-106 P0) — the single contract the UI reads.

Identical shape for live jobs and precomputed showcase trips, so both render through the same
components. Carries EVERY photo (picks + dropped-with-reason), honoring the show-everything principle:
curation is a proposal, not a verdict (spec-107 set_aside generalized to selection).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

__all__ = ["ScoreBreakdown", "Pick", "Dropped", "Person", "Counts", "Album",
           "PICK", "NEAR_DUP", "NOT_SELECTED", "SET_ASIDE"]

# drop-reason codes (a pick has no drop reason)
PICK = "pick"
NEAR_DUP = "near_dup"          # same moment as a kept pick — "a stronger shot was kept"
NOT_SELECTED = "not_selected"  # a survivor that wasn't among the top picks for its scene
SET_ASIDE = "set_aside"        # triage junk (screenshot / document / etc.) — spec-107


@dataclass
class ScoreBreakdown:
    composite: Optional[float] = None
    sharpness: Optional[float] = None     # iqa
    aesthetic: Optional[float] = None     # ava
    occlusion: Optional[float] = None
    tilt: Optional[float] = None


@dataclass
class Pick:
    id: str                               # photo stem (stable across straighten derivatives)
    order: int                            # chronological position in the album
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    scene_cluster: int = -1
    beat: List[str] = field(default_factory=list)   # near-dup ids this pick beat in its moment
    people: List[str] = field(default_factory=list)
    caption: str = ""


@dataclass
class Dropped:
    id: str
    reason: str                           # NEAR_DUP | NOT_SELECTED | SET_ASIDE
    reason_detail: str = ""
    beaten_by: List[str] = field(default_factory=list)  # for NEAR_DUP: the pick(s) that won
    promotable: bool = True               # show-everything: one click back into the album


@dataclass
class Person:
    id: str
    face_count: int = 0
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class Counts:
    uploaded: int = 0
    kept: int = 0
    people: int = 0
    scenes: int = 0
    set_aside: int = 0


@dataclass
class Album:
    job_id: str
    created_at: str
    counts: Counts = field(default_factory=Counts)
    picks: List[Pick] = field(default_factory=list)
    dropped: List[Dropped] = field(default_factory=list)
    people: List[Person] = field(default_factory=list)
    theme: Dict = field(default_factory=dict)      # spec-108 album_theme

    def to_dict(self) -> Dict:
        return asdict(self)
