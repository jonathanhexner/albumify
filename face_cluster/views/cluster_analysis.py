"""spec-045 — typed dataclasses + Service for the Cluster Analysis tab.

Phase 2 ships :class:`ForceMergeResult` (consumed by the Repository's
``save_manual_merge_snapshot`` mutation method). Phases 3–5 add
:class:`ForceMergePreview` and :class:`ClusterAnalysisService`.

Service contract (per spec §6.1):

* ``__init__(repo)`` — owns a :class:`ClusterAnalysisRepository`.
* ``list_clusters()`` / ``get_cluster_ids()`` — cheap passthroughs.
* ``compute_detail_async(cluster_id)`` /
  ``compute_debug_async(cluster_id)`` — heavy compute behind
  :class:`AsyncHandle`; cancels in-flight handles of the same kind.
* ``preview_force_merge(a, b)`` / ``apply_force_merge(a, b, *, merge_round)``
  — typed force-merge.

No Streamlit imports; no ``st.session_state`` writes. Cancellation contract
matches spec §6.1 — caller must inspect handle state before reading
``result``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from face_cluster.views._async import AsyncHandle
from face_cluster.views._base import ClusterRow, _embeddings_matrix, _pairwise_distances
from face_cluster.views._specs import ColumnSpec
from face_cluster.views.cluster_debug_view import ClusterDebugView
from face_cluster.views.cluster_view import ClusterView

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClusterSummaryRow:
    """One row of the all-clusters summary table (spec-074).

    The cheap fields mirror :class:`ClusterRow`; the ``nearest_*`` fields are
    computed across ALL clusters in one pass (``ClusterAnalysisService
    .cluster_summary``) — unlike ``ClusterRow`` where they're placeholders
    until a single cluster is selected.
    """
    cluster_id: int
    size: int
    diameter: float
    avg_intra_dist: float
    n_exemplars: int
    nearest_cluster_id: int        # -1 when there is no other cluster
    nearest_cluster_dist: float    # min exemplar-to-exemplar cosine distance
    nearest_cluster_size: int      # size of that nearest cluster
    merge_candidate: bool          # nearest_cluster_dist < merge_candidate_threshold


# spec-074 — declarative columns for the summary table (raw values read for
# numeric sorting; formatters available for any future strip rendering).
CLUSTER_SUMMARY_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("cluster_id", "Cluster"),
    ColumnSpec("size", "Faces"),
    ColumnSpec("diameter", "Diameter", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("avg_intra_dist", "Avg intra", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("n_exemplars", "Exemplars"),
    ColumnSpec("nearest_cluster_id", "Nearest",
               formatter=lambda v: "-" if v is None or v < 0 else f"C{v}"),
    ColumnSpec("nearest_cluster_dist", "Dist to nearest", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("nearest_cluster_size", "Nearest faces"),
    ColumnSpec("merge_candidate", "Merge?", formatter=lambda v: "yes" if v else ""),
]


@dataclass(frozen=True, slots=True)
class NearestPairRow:
    """One cluster pair, ranked by exemplar distance (spec-075).

    ``evaluated`` is True when the pipeline recorded a merge_decision for this
    pair (i.e. it crossed the candidate threshold); ``rejection_reason`` then
    explains why it wasn't merged. Pairs that were never close enough to
    evaluate still appear here (with evaluated=False) so the user sees what was
    *almost* a merge."""
    cluster_a: int
    cluster_b: int
    size_a: int
    size_b: int
    exemplar_dist: float
    evaluated: bool
    merged: bool
    rejection_reason: Optional[str]


NEAREST_PAIR_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("cluster_a", "A"),
    ColumnSpec("cluster_b", "B"),
    ColumnSpec("size_a", "A faces"),
    ColumnSpec("size_b", "B faces"),
    ColumnSpec("exemplar_dist", "Exemplar dist", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("evaluated", "Evaluated?", formatter=lambda v: "yes" if v else ""),
    ColumnSpec("merged", "Merged?", formatter=lambda v: "yes" if v else ""),
    ColumnSpec("rejection_reason", "Why not merged"),
]


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ForceMergeResult:
    """Outcome of a user-confirmed force-merge.

    Returned by :meth:`ClusterAnalysisRepository.save_manual_merge_snapshot`
    (Phase 2) and surfaced unchanged by
    :meth:`ClusterAnalysisService.apply_force_merge` (Phase 5).
    """
    snapshot_dir: Path
    merge_round: int
    parent_run_dir: Path
    new_cluster_id: int
    n_merged: int


@dataclass(frozen=True, slots=True)
class ForceMergePreview:
    """Typed preview returned by :meth:`ClusterAnalysisService.preview_force_merge`.

    Field semantics mirror the legacy ``_compute_force_merge_preview`` dict
    (legacy ``cluster_analysis_tab.py`` line ~99), but with explicit field
    names instead of magic-string keys. The three gates are independent —
    the UI badges each separately.
    """
    cluster_a: int
    cluster_b: int
    exemplar_dist: float
    threshold: float
    is_candidate: bool
    passes_exemplar: bool
    passes_support: bool
    passes_diameter: bool
    n_gates_passed: int
    post_diameter: float
    support: int
    cluster_a_size: int
    cluster_b_size: int
    exemplar_face_ids_a: List[int]
    exemplar_face_ids_b: List[int]


# ---------------------------------------------------------------------------
# Service (spec §6.1)
# ---------------------------------------------------------------------------

class ClusterAnalysisService:
    """Typed read + compute API for the Cluster Analysis tab.

    Owns a single :class:`ClusterAnalysisRepository`. Stateless across
    methods *except* for two slots that remember the in-flight async
    handles per compute kind — so a fresh ``compute_*_async`` call can
    cancel the prior one (spec §6.1).
    """

    def __init__(self, repo) -> None:
        if repo is None:
            raise ValueError("ClusterAnalysisService requires a non-None Repository.")
        self._repo = repo
        self._detail_handle: Optional[AsyncHandle[ClusterView]] = None
        self._debug_handle: Optional[AsyncHandle[ClusterDebugView]] = None

    # ---- Phase 3: cheap reads -----------------------------------------

    def list_clusters(self) -> List[ClusterRow]:
        """Return one :class:`ClusterRow` per real cluster (passthrough)."""
        return self._repo.get_cluster_rows()

    def get_cluster_ids(self) -> List[int]:
        """Cluster ids in display order (passthrough)."""
        return self._repo.get_cluster_ids()

    def cluster_summary(self) -> List[ClusterSummaryRow]:
        """All-clusters overview: size/diameter/spread + nearest other cluster.

        Cheap fields come from ``get_cluster_rows``; the nearest-cluster id /
        distance / size are computed across every cluster pair in one pass
        (min exemplar-to-exemplar cosine distance on normalized embeddings).
        Sub-second for typical run sizes — the v2 tab caches it per run dir.
        """
        rows = self._repo.get_cluster_rows()
        if not rows:
            return []
        size_by_id = {r.cluster_id: r.size for r in rows}
        ex_mat = self._exemplar_matrices(rows)

        meta = self._repo.get_run_metadata()
        threshold = float(meta.config.get("merge_candidate_threshold", 0.45))

        out: List[ClusterSummaryRow] = []
        for r in rows:
            cid = r.cluster_id
            nid, ndist = -1, 1.0
            a = ex_mat.get(cid)
            if a is not None:
                for ocid, b in ex_mat.items():
                    if ocid == cid:
                        continue
                    d = float(1.0 - (a @ b.T).max())  # min cosine dist (normalized embs)
                    if d < ndist:
                        ndist, nid = d, ocid
            out.append(ClusterSummaryRow(
                cluster_id=cid, size=r.size, diameter=r.diameter,
                avg_intra_dist=r.avg_intra_dist, n_exemplars=r.n_exemplars,
                nearest_cluster_id=nid, nearest_cluster_dist=round(ndist, 4),
                nearest_cluster_size=size_by_id.get(nid, 0),
                merge_candidate=(nid >= 0 and ndist < threshold),
            ))
        return out

    def nearest_cluster_pairs(self, top_n: int = 20) -> List[NearestPairRow]:
        """The ``top_n`` closest cluster pairs by exemplar distance (spec-075).

        Each pair joins its ``merge_decisions`` verdict when one exists, so the
        user can see "what was almost a merge, and why it wasn't" — including
        pairs that never crossed the candidate threshold (evaluated=False)."""
        rows = self._repo.get_cluster_rows()
        if len(rows) < 2:
            return []
        size_by_id = {r.cluster_id: r.size for r in rows}
        ex_mat = self._exemplar_matrices(rows)
        merges = {}
        for m in self._repo.get_merge_log():
            merges[frozenset((m.cluster_a, m.cluster_b))] = m

        ids = [r.cluster_id for r in rows]
        pairs: List[NearestPairRow] = []
        for i, a in enumerate(ids):
            if a not in ex_mat:
                continue
            for b in ids[i + 1:]:
                if b not in ex_mat:
                    continue
                d = float(1.0 - (ex_mat[a] @ ex_mat[b].T).max())
                m = merges.get(frozenset((a, b)))
                pairs.append(NearestPairRow(
                    cluster_a=a, cluster_b=b,
                    size_a=size_by_id[a], size_b=size_by_id[b],
                    exemplar_dist=round(d, 4),
                    evaluated=m is not None,
                    merged=bool(getattr(m, "actually_merged", False)) if m else False,
                    rejection_reason=(getattr(m, "rejection_reason", None) if m else None),
                ))
        pairs.sort(key=lambda p: p.exemplar_dist)
        return pairs[:top_n]

    def _exemplar_matrices(self, rows) -> dict:
        """{cluster_id -> (k x d) exemplar embedding matrix} for all rows.

        Exemplars-first, falling back to the first member when a cluster has no
        exemplar set. Shared by ``cluster_summary`` and ``nearest_cluster_pairs``."""
        proxy = self._build_pipeline_result_proxy()
        cr = proxy.cluster_result
        faces = proxy.faces
        out = {}
        for r in rows:
            cid = r.cluster_id
            idxs = list(cr.exemplars.get(cid) or cr.clusters.get(cid, [])[:1])
            m = _embeddings_matrix(faces, idxs)
            if m is not None:
                out[cid] = m
        return out

    def exemplar_face_ids(self, cluster_id: int, n: int = 8) -> List[int]:
        """Return up to ``n`` representative face_ids for a cluster.

        Exemplars first (the cluster's canonical faces), padded with other
        members if the cluster has fewer than ``n`` exemplars. Cheap: two
        small assignment reads, no distance-matrix compute — this is the
        Gallery strip's data source (spec-066 D1), deliberately lighter than
        the full :meth:`compute_detail`.

        Args:
            cluster_id: the cluster to sample.
            n: max face_ids to return (the Gallery strip width).

        Returns:
            Ordered face_ids, exemplars first, length ``min(n, cluster_size)``.
        """
        from sim_bench.db.face_clustering.cluster_analysis_repo import (
            ClusterAnalysisCriteria,
        )

        if n <= 0:
            return []
        exemplars = self._repo.find_assignments(
            ClusterAnalysisCriteria(cluster_id=cluster_id, exemplars_only=True)
        )
        ordered = [a.face_id for a in exemplars]
        if len(ordered) < n:
            members = self._repo.find_assignments(
                ClusterAnalysisCriteria(cluster_id=cluster_id)
            )
            seen = set(ordered)
            for a in members:
                if a.face_id not in seen:
                    ordered.append(a.face_id)
                    seen.add(a.face_id)
                    if len(ordered) >= n:
                        break
        return ordered[:n]

    def low_quality_face_ids(self, face_ids: List[int]) -> set:
        """Subset of ``face_ids`` that did NOT pass the quality gate.

        Uses the already-computed ``FaceRecord.is_core`` flag (a face in a
        cluster with ``is_core=False`` was *attached* despite being below the
        core-quality bar). Cheap read; the Gallery strip renders a ``warning``
        badge for these (spec-066 G5-flag, read-only — disqualify is spec-069).
        """
        if not face_ids:
            return set()
        records = self._repo.get_face_records(list(face_ids))
        return {r.face_id for r in records if not getattr(r, "is_core", True)}

    # ---- Synchronous compute (used by the v2 tab; SIGHTING-079 fix) ---

    def compute_detail(self, cluster_id: int) -> ClusterView:
        """Synchronously compute :class:`ClusterView` for ``cluster_id``.

        Streamlit's request/response model doesn't poll background threads,
        so AsyncHandle doesn't work as a UI primitive without an explicit
        ``time.sleep + st.rerun`` loop in the caller. For a typical cluster
        (≤100 faces) compute is sub-second — sync + ``st.spinner`` in the
        caller is simpler and matches the Streamlit lifecycle. SIGHTING-079.
        """
        result_proxy = self._build_pipeline_result_proxy()
        return ClusterView.compute(result_proxy, cluster_id)

    def compute_debug(self, cluster_id: int) -> ClusterDebugView:
        """Synchronously compute :class:`ClusterDebugView` for ``cluster_id``.

        Same rationale as :meth:`compute_detail` — sync fits Streamlit.
        """
        result_proxy = self._build_pipeline_result_proxy()
        return ClusterDebugView.compute(result_proxy, cluster_id)

    # ---- Async compute (kept as a library primitive for future tabs
    #      with heavy compute that genuinely need background work +
    #      polling. NOT used by the v2 Cluster Analysis tab — see
    #      SIGHTING-079 for why.) ----------------------------------------

    def compute_detail_async(self, cluster_id: int) -> AsyncHandle[ClusterView]:
        """Start a background compute of :class:`ClusterView` for ``cluster_id``.

        Cancels any in-flight handle on this Service instance before starting.
        Caller is responsible for polling the returned handle AND triggering
        re-renders (e.g., ``time.sleep + st.rerun()``) until ``state == "done"``.

        For the v2 Cluster Analysis tab, prefer :meth:`compute_detail` — sync
        compute matches Streamlit's lifecycle. This async variant remains for
        future tabs that need true backgrounding.
        """
        if self._detail_handle is not None and self._detail_handle.state in ("pending", "running"):
            self._detail_handle.cancel()
        result_proxy = self._build_pipeline_result_proxy()
        self._detail_handle = AsyncHandle.start(ClusterView.compute, result_proxy, cluster_id)
        return self._detail_handle

    def compute_debug_async(self, cluster_id: int) -> AsyncHandle[ClusterDebugView]:
        """Async variant of :meth:`compute_debug`. See ``compute_detail_async``."""
        if self._debug_handle is not None and self._debug_handle.state in ("pending", "running"):
            self._debug_handle.cancel()
        result_proxy = self._build_pipeline_result_proxy()
        self._debug_handle = AsyncHandle.start(ClusterDebugView.compute, result_proxy, cluster_id)
        return self._debug_handle

    # ---- Phase 5: force merge ------------------------------------------

    def preview_force_merge(self, cluster_a: int, cluster_b: int) -> ForceMergePreview:
        """Compute the typed gate-by-gate preview for a candidate merge.

        Pure compute; no FS / no session_state. Mirrors the legacy
        ``_compute_force_merge_preview`` body — three gates (exemplar /
        support / post-diameter) against the merge_candidate_threshold
        from run metadata.
        """
        from face_cluster.features import FeatureComputer, MergeFeatureContext

        meta = self._repo.get_run_metadata()
        threshold = float(meta.config.get("merge_candidate_threshold", 0.45))
        result_proxy = self._build_pipeline_result_proxy()
        cr = result_proxy.cluster_result
        if cluster_a not in cr.clusters or cluster_b not in cr.clusters:
            raise ValueError(
                f"unknown cluster id(s): a={cluster_a}, b={cluster_b}; "
                f"known={sorted(cr.clusters.keys())}"
            )

        dm = self._distance_matrix(result_proxy)
        ctx = MergeFeatureContext(cluster_result=cr, faces=result_proxy.faces, distance_matrix=dm)
        fc = FeatureComputer()
        t_global = fc._compute_t_global(cr, dm)
        feat = fc.compute_pair_features(cluster_a, cluster_b, ctx, t_global)

        ex_a = cr.exemplars.get(cluster_a, cr.clusters[cluster_a])
        ex_b = cr.exemplars.get(cluster_b, cr.clusters[cluster_b])
        exemplar_dist = float(dm[ex_a[0], ex_b[0]]) if ex_a and ex_b else 1.0
        passes_exemplar = exemplar_dist <= threshold
        passes_support = (feat.n_cross_pairs_below_threshold or 0) >= 1
        passes_diameter = (feat.post_merge_diameter or 0) <= (
            max(feat.diameter_a or 0, feat.diameter_b or 0) * 1.5 + 0.05
        )
        return ForceMergePreview(
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            exemplar_dist=exemplar_dist,
            threshold=threshold,
            is_candidate=exemplar_dist <= threshold,
            passes_exemplar=passes_exemplar,
            passes_support=passes_support,
            passes_diameter=passes_diameter,
            n_gates_passed=sum([passes_exemplar, passes_support, passes_diameter]),
            post_diameter=float(feat.post_merge_diameter or 0.0),
            support=int(feat.n_cross_pairs_below_threshold or 0),
            cluster_a_size=len(cr.clusters[cluster_a]),
            cluster_b_size=len(cr.clusters[cluster_b]),
            exemplar_face_ids_a=[result_proxy.faces[i].face_id for i in ex_a[:3]],
            exemplar_face_ids_b=[result_proxy.faces[i].face_id for i in ex_b[:3]],
        )

    def apply_force_merge(
        self, cluster_a: int, cluster_b: int, *, merge_round: int
    ) -> ForceMergeResult:
        """Persist a force-merge as a fresh snapshot dir. Delegates to the
        Repository; no session_state touched here.

        Uses a default-constructed legacy :class:`PipelineConfig` for the
        snapshot's audit copy — production callers should pass the run's
        actual config once the v2 typed config story lands (out of scope
        for spec-045)."""
        from face_cluster.config import PipelineConfig

        return self._repo.save_manual_merge_snapshot(
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            merge_round=merge_round,
            config=PipelineConfig(),
        )

    # ---- internals ------------------------------------------------------

    def _build_pipeline_result_proxy(self):
        """Assemble the minimal :class:`PipelineResult` the legacy
        ``ClusterView.compute`` / ``ClusterDebugView.compute`` classmethods
        consume. See spec D5 — the typed-input refactor of those classmethods
        is deferred.

        Strips the noise cluster (``NOISE_LABEL``) from the ClusterResult
        before passing it on. Without this, ``ClusterView.compute`` crashes
        when it tries to ``np.stack`` the exemplar embeddings of a noise
        cluster (it has none). The Repository already excludes noise from
        ``get_cluster_rows`` — this keeps the Service's compute path
        consistent with that contract.
        """
        from dataclasses import replace

        from face_cluster.pipeline import PipelineResult
        from sim_bench.pipeline.clustering_labels import is_noise

        cr = self._repo.get_cluster_result("final")
        clean_clusters = {cid: idxs for cid, idxs in cr.clusters.items() if not is_noise(cid)}
        clean_exemplars = {cid: ex for cid, ex in cr.exemplars.items() if not is_noise(cid)}
        clean_cr = replace(cr, clusters=clean_clusters, exemplars=clean_exemplars)

        return PipelineResult(
            faces=self._repo._run_store.faces(),
            cluster_result=clean_cr,
            output_dir=self._repo._config.run_dir,
            summary={"config": self._repo.get_run_metadata().config},
        )

    def _distance_matrix(self, result):
        """Pairwise distance matrix over ``result.faces`` (n × n). Returns
        None when fewer than 2 faces have embeddings.

        Thin wrapper around the shared helpers in ``views._base`` (DUP-1
        resolved 2026-05-29 — was a 13-LOC duplicate of those primitives)."""
        faces = result.faces
        if len(faces) < 2:
            return None
        mat = _embeddings_matrix(faces, list(range(len(faces))))
        return _pairwise_distances(mat) if mat is not None else None


__all__ = [
    "ForceMergeResult",
    "ForceMergePreview",
    "ClusterSummaryRow",
    "CLUSTER_SUMMARY_COLUMNS",
    "NearestPairRow",
    "NEAREST_PAIR_COLUMNS",
    "ClusterAnalysisService",
]
