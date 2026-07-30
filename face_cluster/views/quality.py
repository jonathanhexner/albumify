"""spec-065 — Service for the v2 Quality tab.

Read-only viewer over the per-run ``filter_decisions`` table. Aggregates
per-gate pass/fail counts so an operator can answer "with profile X, how
many faces got rejected on blur?".

Architecture rules (mirrors spec-063 / spec-064):

* Streamlit-free — no ``streamlit`` imports here.
* Sync compute — a single SELECT + a Python aggregation pass; SIGHTING-079
  said no background polling.
* Typed I/O — returns ``QualitySummary`` / ``list[FilterDecisionRow]``;
  never raw dicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepository,
    FilterDecisionCriteria,
)
from sim_bench.run_db.store import FilterDecisionRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GateCounts:
    """Pass / fail counts for one gate (e.g. "blur")."""
    gate_name: str
    n_passed: int
    n_rejected: int

    @property
    def n_total(self) -> int:
        """Total decisions for this gate (passed + rejected)."""
        return self.n_passed + self.n_rejected


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """Aggregated per-gate verdicts across all items in the run.

    ``gates`` is sorted by ``gate_name`` for stable display ordering.
    ``top_rejection_gate`` is the gate that rejected the most items;
    ``None`` if no rejections occurred.
    """
    n_items: int                       # distinct item_ids that had ≥ 1 decision
    n_decisions: int                   # total filter_decisions rows
    n_rejected: int                    # total rows with rejected=True
    top_rejection_gate: Optional[str]
    gates: List[GateCounts]

    @property
    def pass_rate(self) -> float:
        """Overall fraction of decisions that passed; 0.0 when no decisions."""
        if self.n_decisions == 0:
            return 0.0
        return 1.0 - (self.n_rejected / self.n_decisions)


class QualityService:
    """Typed read API for the v2 Quality tab."""

    def __init__(self, repo: ClusterAnalysisRepository) -> None:
        if repo is None:
            raise ValueError(
                "QualityService requires a non-None ClusterAnalysisRepository."
            )
        self._repo = repo

    def list_filter_decisions(
        self,
        criteria: Optional[FilterDecisionCriteria] = None,
    ) -> List[FilterDecisionRow]:
        """Pass-through to the Repository; ``None`` = unfiltered."""
        return self._repo.list_filter_decisions(criteria or FilterDecisionCriteria())

    def summary(self) -> QualitySummary:
        """Compute aggregated per-gate pass/fail counts across all decisions."""
        rows = self._repo.list_filter_decisions(FilterDecisionCriteria())
        passed: Dict[str, int] = {}
        rejected: Dict[str, int] = {}
        items: set[str] = set()
        for r in rows:
            items.add(r.item_id)
            bucket = rejected if r.rejected else passed
            bucket[r.filter_name] = bucket.get(r.filter_name, 0) + 1

        gate_names = sorted(set(passed) | set(rejected))
        gates = [
            GateCounts(
                gate_name=name,
                n_passed=passed.get(name, 0),
                n_rejected=rejected.get(name, 0),
            )
            for name in gate_names
        ]
        top_gate: Optional[str] = None
        if rejected:
            top_gate = max(rejected.items(), key=lambda kv: kv[1])[0]
        return QualitySummary(
            n_items=len(items),
            n_decisions=len(rows),
            n_rejected=sum(1 for r in rows if r.rejected),
            top_rejection_gate=top_gate,
            gates=gates,
        )


__all__ = ["QualityService", "QualitySummary", "GateCounts"]
