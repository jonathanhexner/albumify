"""Shared vocabulary for the AI verification layer (spec-111 P0).

Every checkpoint (verify_face_clusters, verify_selection, verify_scenes, verify_triage) produces a
``VerifyResult`` of ``Correction``s, and is gated by an ``EvalResult`` (must beat the deterministic
baseline on a human-labeled set before it ships). Pure dataclasses, no framework coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["Correction", "VerifyResult", "EvalResult",
           "MERGE", "SPLIT", "PROMOTE", "DEMOTE", "UNSET_ASIDE", "REASSIGN"]

# correction kinds (what a verifier decided to change)
MERGE = "merge"            # two face/scene clusters are actually one
SPLIT = "split"            # one cluster is actually two
PROMOTE = "promote"        # add a photo into the album (soft-K standout)
DEMOTE = "demote"          # drop a weak pick from the album
UNSET_ASIDE = "unset_aside"  # a real photo was wrongly triaged out
REASSIGN = "reassign"      # move a photo to a different cluster / better best-of-cluster


@dataclass
class Correction:
    """One change a verifier wants applied forward into the context — surfaced + reversible."""
    kind: str
    targets: List[str]           # photo / cluster ids affected
    reason: str                  # human-readable (shown in the album: "promoted because…")
    detail: Dict = field(default_factory=dict)  # kind-specific payload


@dataclass
class VerifyResult:
    """What a checkpoint returns after reviewing a set of decisions."""
    checkpoint: str
    n_reviewed: int                               # decisions inspected
    escalated: List[str] = field(default_factory=list)   # ids sent to the VLM (borderline subset)
    corrections: List[Correction] = field(default_factory=list)

    @property
    def n_escalated(self) -> int:
        return len(self.escalated)

    @property
    def n_corrections(self) -> int:
        return len(self.corrections)


@dataclass
class EvalResult:
    """AC1 gate: does the verifier beat the deterministic baseline on human-labeled data?"""
    checkpoint: str
    n: int                       # labeled items evaluated
    baseline_agreement: float    # fraction where baseline == human truth
    verified_agreement: float    # fraction where baseline+verifier == human truth
    n_changed: int               # items the verifier changed from baseline
    helped: int                  # changed from wrong → right
    hurt: int                    # changed from right → wrong

    @property
    def delta(self) -> float:
        return round(self.verified_agreement - self.baseline_agreement, 4)

    @property
    def improved(self) -> bool:
        """Ships only if it raises agreement AND doesn't harm more than it helps."""
        return self.verified_agreement > self.baseline_agreement and self.helped >= self.hurt
