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

from face_cluster.run_layout import crop_path
from face_cluster.types import MergeDecisionRow
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisCriteria,
    ClusterAnalysisRepository,
)

logger = logging.getLogger(__name__)

# spec-071: the 4-5 gates the merger evaluates, in display order.
_GATE_ORDER = ("cross", "exemplar", "support", "margin", "diameter")


def _fmt(v: float) -> str:
    """Compact float, with inf rendered as a word (margin gate can be inf)."""
    if v is None:
        return "?"
    if v == float("inf"):
        return "inf"
    return f"{float(v):.3f}"


@dataclass(frozen=True, slots=True)
class GateBadge:
    """One merge gate's verdict for a pair (spec-071)."""
    name: str
    passed: Optional[bool]   # None = gate not evaluated / disabled
    detail: str              # measured value vs threshold, human-readable


@dataclass(frozen=True, slots=True)
class PairFaces:
    """Faces of the two clusters in a merge decision (spec-071).

    Resolved against the persisted clustering (the only iteration stored in
    ``cluster_assignments``). The merger keeps the min cluster id on merge, so
    a decision's ``cluster_a/cluster_b`` are base cluster ids that resolve
    here. Intermediate merge-round states are not persisted, so for multi-round
    merges these faces approximate the cluster at the persisted iteration.
    """
    cluster_a: int
    cluster_b: int
    a_face_ids: List[int]
    b_face_ids: List[int]
    a_crops: List[str]
    b_crops: List[str]


@dataclass(frozen=True, slots=True)
class MergeReviewSummary:
    """Top-of-tab merge summary (spec-071)."""
    n_considered: int
    n_merged: int
    n_rejected: int
    top_rejection_gate: Optional[str]


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

    # ------------------------------------------------------------------ spec-071

    def gate_badges(self, row: MergeDecisionRow) -> List[GateBadge]:
        """Per-gate verdict for one decision (cross / exemplar / support /
        margin / diameter), each with a value-vs-threshold detail string.
        Pure mapping of fields already on ``MergeDecisionRow`` — no I/O."""
        return [
            GateBadge("cross", row.passes_cross,
                      f"p25={_fmt(row.p25_cross_dist)} vs T_global={_fmt(row.T_global)}"),
            GateBadge("exemplar", row.passes_exemplar,
                      f"dist {_fmt(row.exemplar_dist)} <= {_fmt(row.threshold_used)}"),
            GateBadge("support", row.passes_support,
                      f"{row.support}/{row.required_support}"),
            GateBadge("margin", row.passes_margin,
                      ("gate off" if row.margin_gap == float("inf")
                       else f"gap {_fmt(row.margin_gap)} (vs cluster {row.margin_competitor_id})")),
            GateBadge("diameter", row.passes_diameter,
                      f"{_fmt(row.post_diameter)} <= {_fmt(row.max_allowed_diameter)}"),
        ]

    def summary(self) -> MergeReviewSummary:
        """Counts for the top strip: merges executed, pairs rejected, and which
        gate failed most often among rejected pairs."""
        rows = self._repo.get_merge_log()
        n_merged = sum(1 for r in rows if r.actually_merged)
        rejected = [r for r in rows if not r.actually_merged and r.action == "rejected"]
        fail_counts: dict[str, int] = {}
        for r in rejected:
            for b in self.gate_badges(r):
                if b.passed is False:
                    fail_counts[b.name] = fail_counts.get(b.name, 0) + 1
        top = max(fail_counts, key=fail_counts.get) if fail_counts else None
        return MergeReviewSummary(
            n_considered=len(rows), n_merged=n_merged,
            n_rejected=len(rejected), top_rejection_gate=top,
        )

    def pair_faces(self, row: MergeDecisionRow, max_per_cluster: int = 24) -> PairFaces:
        """Face ids + crop paths for cluster_a and cluster_b of a decision.

        Resolved against the persisted clustering (see :class:`PairFaces` note).
        Caps each side at ``max_per_cluster`` thumbnails.
        """
        run_dir = self._repo._config.run_dir
        assigns = self._repo.find_assignments(ClusterAnalysisCriteria())
        a_ids = sorted(a.face_id for a in assigns if a.cluster_id == row.cluster_a)
        b_ids = sorted(a.face_id for a in assigns if a.cluster_id == row.cluster_b)

        def crops(ids: List[int]) -> List[str]:
            out: List[str] = []
            for fid in ids[:max_per_cluster]:
                p = crop_path(run_dir, fid)
                if p.is_file():
                    out.append(str(p))
            return out

        return PairFaces(
            cluster_a=row.cluster_a, cluster_b=row.cluster_b,
            a_face_ids=a_ids, b_face_ids=b_ids,
            a_crops=crops(a_ids), b_crops=crops(b_ids),
        )


__all__ = [
    "MergedClustersService", "MergeDecisionCriteria",
    "GateBadge", "PairFaces", "MergeReviewSummary",
]
