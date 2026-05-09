"""RunOverview: full-run health check view."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from face_cluster.pipeline import PipelineResult
from face_cluster.views._base import (
    ClusterRow, _embeddings_matrix, _pairwise_distances,
)

logger = logging.getLogger(__name__)


@dataclass
class RunOverview:
    """Full-run health check for the Run Overview tab."""

    # Quality gate funnel
    n_images: int
    n_faces_detected: int
    n_core: int
    n_holdout: int

    # Clustering summary
    n_clusters: int
    n_noise: int
    cluster_rows: List[ClusterRow]

    # UMAP (None if fewer than 5 core faces)
    umap_coords: Optional[np.ndarray]   # shape (n_core, 2)
    umap_labels: Optional[np.ndarray]   # cluster_id per core face (-1 = noise)
    umap_face_ids: Optional[List[int]]  # face_id per UMAP point

    # Stage timing from pipeline_run.json (may be empty)
    stage_timings: Dict[str, float]

    @classmethod
    def compute(cls, result: PipelineResult) -> "RunOverview":
        faces = result.faces
        cr = result.cluster_result
        summary = result.summary
        n_core = summary.get("n_core", 0)
        n_noise = summary.get("n_noise", 0)

        # Cluster rows — need embeddings to compute distances
        cluster_rows: List[ClusterRow] = []
        all_cluster_ids = sorted(cr.clusters.keys())

        # Build per-cluster exemplar embedding vectors for nearest-cluster computation
        exemplar_embeds: Dict[int, np.ndarray] = {}
        for cid, member_indices in cr.clusters.items():
            ex_indices = cr.exemplars.get(cid, member_indices[:1])
            mat = _embeddings_matrix(faces, ex_indices)
            if mat is not None:
                exemplar_embeds[cid] = mat.mean(axis=0)

        # Merge threshold from run config (not hardcoded)
        run_cfg = summary.get("config") or {}
        merge_threshold = float(run_cfg.get("merge_candidate_threshold", 0.45))

        for cid in all_cluster_ids:
            member_indices = cr.clusters[cid]
            stats = cr.cluster_stats.get(cid, {})
            mat = _embeddings_matrix(faces, member_indices)

            if mat is not None and len(mat) > 1:
                pw = _pairwise_distances(mat)
                diameter = float(pw.max())
                avg_intra = float(pw[np.triu_indices_from(pw, k=1)].mean())
            elif mat is not None and len(mat) == 1:
                diameter = 0.0
                avg_intra = 0.0
            else:
                diameter = float(stats.get("diameter", 0.0))
                avg_intra = float(stats.get("avg_dist", 0.0))

            # Nearest cluster by exemplar distance
            nearest_cid = -1
            nearest_dist = float("inf")
            my_ex = exemplar_embeds.get(cid)
            if my_ex is not None:
                for other_cid, other_ex in exemplar_embeds.items():
                    if other_cid == cid:
                        continue
                    d = float(np.clip(1.0 - float(my_ex @ other_ex), 0.0, 2.0))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_cid = other_cid

            ex_ids = [faces[i].face_id for i in cr.exemplars.get(cid, [])]
            cluster_rows.append(ClusterRow(
                cluster_id=cid,
                size=len(member_indices),
                diameter=round(diameter, 4),
                avg_intra_dist=round(avg_intra, 4),
                n_exemplars=len(ex_ids),
                nearest_cluster_id=nearest_cid,
                nearest_cluster_dist=round(min(nearest_dist, 9.999), 4),
                merge_candidate=(nearest_dist < merge_threshold),
            ))

        # UMAP
        umap_coords = None
        umap_labels = None
        umap_face_ids = None
        core_indices = [i for i, f in enumerate(faces) if f.is_core]
        if len(core_indices) >= 5:
            mat = _embeddings_matrix(faces, core_indices)
            if mat is not None:
                try:
                    import umap as umap_lib
                    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
                    umap_coords = reducer.fit_transform(mat).astype(np.float32)
                    umap_labels = np.array([cr.labels[i] if i < len(cr.labels) else -1 for i in range(len(core_indices))], dtype=np.int32)
                    umap_face_ids = [faces[i].face_id for i in core_indices]
                except Exception as exc:
                    logger.warning(f"UMAP failed: {exc}")

        # Stage timings from summary
        stage_timings = summary.get("stages_timing", {}) or {}

        n_images = len({f.image_path for f in faces if f.image_path})

        return cls(
            n_images=n_images,
            n_faces_detected=len(faces),
            n_core=n_core,
            n_holdout=len(faces) - n_core,
            n_clusters=cr.n_clusters,
            n_noise=n_noise,
            cluster_rows=cluster_rows,
            umap_coords=umap_coords,
            umap_labels=umap_labels,
            umap_face_ids=umap_face_ids,
            stage_timings=stage_timings,
        )
