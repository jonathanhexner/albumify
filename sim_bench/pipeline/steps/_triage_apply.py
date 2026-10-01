"""Shared triage apply-helper (spec-107): record decisions + filter survivors.

Framework-coupled (touches PipelineContext), so it lives on the step side, not in the
``sim_bench.triage`` domain module (which stays framework-agnostic per spec-053).
``set_aside`` images are removed from the curation set; ``review`` images are kept but flagged;
``keep`` passes through. Nothing is deleted.
"""
from __future__ import annotations

from typing import Dict

from sim_bench.pipeline.context import PipelineContext
from sim_bench.triage.metadata import TriageResult


def partition_and_record(context: PipelineContext, decisions: Dict[str, TriageResult], tier: str) -> int:
    """Apply per-image triage decisions to the context. Returns the count set aside.

    - ``set_aside`` → recorded in ``context.set_aside`` and removed from ``image_paths``.
    - ``review``    → recorded in ``context.triage_review``, kept in ``image_paths`` (flagged).
    - ``keep`` / missing → pass through unchanged.
    """
    survivors = []
    n_aside = 0
    for p in context.image_paths:
        r = decisions.get(str(p))
        if r and r.decision == "set_aside":
            context.set_aside.append({"image": str(p), "reason": r.reason,
                                      "confidence": r.confidence, "tier": tier})
            n_aside += 1
            continue
        if r and r.decision == "review":
            context.triage_review.append({"image": str(p), "reason": r.reason,
                                          "confidence": r.confidence, "tier": tier})
        survivors.append(p)
    context.image_paths = survivors
    context.active_images = {str(p) for p in survivors}
    return n_aside
