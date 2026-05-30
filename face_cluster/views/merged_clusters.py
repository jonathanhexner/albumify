"""spec-065 — Service for the v2 Merged Clusters tab.

Read-only viewer over the per-run ``merge_decisions`` table. The merger
writes one row per candidate pair it considered (whether it merged or
not); this Service exposes that history with simple filters so an
operator can answer "why did A and B merge?" / "why didn't C and D?".

Architecture rules (mirrors spec-063 / spec-064):

* Streamlit-free — no ``streamlit`` imports here.
* Sync compute — the read is a single SELECT over a small table
  (typically << 1000 rows); SIGHTING-079 said no background polling.
* Typed I/O — returns ``list[MergeDecisionRow]``; never raw dicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from face_cluster.types import MergeDecisionRow
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MergeDecisionCriteria:
    """Filter for :meth:`MergedClustersService.list_merge_decisions`.

    All fields optional; ``None`` = unfiltered. Filtering is performed
    in-process because the merge log is bounded by O(n_clusters^2) per
    iteration and is small in practice — a Python filter avoids a second
    DB round-trip versus SQL pushdown.
    """
    actually_merged: Optional[bool] = None    # True = only executed merges
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    iteration: Optional[int] = None


class MergedClustersService:
    """Typed read API for the v2 Merged Clusters tab."""

    def __init__(self, repo: ClusterAnalysisRepository) -> None:
        if repo is None:
            raise ValueError(
                "MergedClustersService requires a non-None ClusterAnalysisRepository."
            )
        self._repo = repo

    def list_merge_decisions(
        self,
        criteria: Optional[MergeDecisionCriteria] = None,
    ) -> List[MergeDecisionRow]:
        """Return merge_decisions rows matching ``criteria`` (defaults: all)."""
        crit = criteria or MergeDecisionCriteria()
        rows = self._repo.get_merge_log()
        if crit.actually_merged is not None:
            rows = [r for r in rows if r.actually_merged == crit.actually_merged]
        if crit.cluster_a is not None:
            rows = [r for r in rows if r.cluster_a == crit.cluster_a]
        if crit.cluster_b is not None:
            rows = [r for r in rows if r.cluster_b == crit.cluster_b]
        if crit.iteration is not None:
            rows = [r for r in rows if r.iteration == crit.iteration]
        return rows

    def list_iterations(self) -> List[int]:
        """Distinct iteration ids present in the merge log, ascending."""
        return sorted({r.iteration for r in self._repo.get_merge_log()})

    def list_clusters_in_log(self) -> List[int]:
        """Distinct cluster ids that appear as cluster_a OR cluster_b, ascending."""
        rows = self._repo.get_merge_log()
        ids = {r.cluster_a for r in rows} | {r.cluster_b for r in rows}
        return sorted(ids)


__all__ = ["MergedClustersService", "MergeDecisionCriteria"]
