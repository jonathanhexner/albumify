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
from face_cluster.views.cluster_debug_view import ClusterDebugView
from face_cluster.views.cluster_view import ClusterView

logger = logging.getLogger(__name__)


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

    # ---- Phase 4: heavy compute behind AsyncHandle --------------------

    def compute_detail_async(self, cluster_id: int) -> AsyncHandle[ClusterView]:
        """Start a background compute of :class:`ClusterView` for ``cluster_id``.

        Cancels any in-flight ``compute_detail_async`` handle on this
        Service instance before starting (single-cluster contract per
        spec §6.1). Caller polls the returned handle.
        """
        if self._detail_handle is not None and self._detail_handle.state in ("pending", "running"):
            self._detail_handle.cancel()
        result_proxy = self._build_pipeline_result_proxy()
        self._detail_handle = AsyncHandle.start(ClusterView.compute, result_proxy, cluster_id)
        return self._detail_handle

    def compute_debug_async(self, cluster_id: int) -> AsyncHandle[ClusterDebugView]:
        """Start a background compute of :class:`ClusterDebugView` for ``cluster_id``.

        Same cancellation contract as :meth:`compute_detail_async`.
        """
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
    "ClusterAnalysisService",
]
