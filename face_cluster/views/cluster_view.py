"""ClusterView: per-cluster drill-down."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from face_cluster.pipeline import PipelineResult
from face_cluster.views._base import (
    FaceRow, NearestClusterRow,
    _embeddings_matrix, _pairwise_distances, _exemplar_dist, _dist_to_centroid, _face_row,
)


@dataclass
class ClusterView:
    """Per-cluster drill-down for the Cluster Analysis tab."""

    cluster_id: int
    size: int
    diameter: float
    avg_intra_dist: float

    exemplar_face_ids: List[int]
    faces: List[FaceRow]          # all members, sorted by dist_to_exemplar
    outlier_face_ids: List[int]   # dist_to_exemplar > p90
    split_signal: bool            # bimodal gap detected in intra-cluster distances

    nearest_clusters: List[NearestClusterRow]

    @classmethod
    def compute(cls, result: PipelineResult, cluster_id: int) -> "ClusterView":
        faces = result.faces
        cr = result.cluster_result

        if cluster_id not in cr.clusters:
            raise ValueError(f"cluster_id {cluster_id} not found in result")

        member_indices = cr.clusters[cluster_id]
        ex_face_ids = [faces[i].face_id for i in cr.exemplars.get(cluster_id, member_indices[:1])]
        ex_local = [j for j, i in enumerate(member_indices)
                    if faces[i].face_id in ex_face_ids]

        mat = _embeddings_matrix(faces, member_indices)
        if mat is not None and len(mat) > 1:
            pw = _pairwise_distances(mat)
            diameter = float(pw.max())
            avg_intra = float(pw[np.triu_indices_from(pw, k=1)].mean())
            d_exemplar = _exemplar_dist(mat, ex_local)
            d_centroid = _dist_to_centroid(mat)
            outlier_threshold = float(np.percentile(d_exemplar, 90)) if len(d_exemplar) > 2 else float(d_exemplar.max())
            # Split signal: large gap (>0.05) in sorted intra-distances suggests bimodal
            upper_dists = pw[np.triu_indices_from(pw, k=1)]
            sorted_d = np.sort(upper_dists)
            gaps = np.diff(sorted_d)
            split_signal = bool(gaps.max() > 0.12 and gaps.max() > 3 * gaps.mean()) if len(gaps) > 0 else False
        else:
            diameter = avg_intra = 0.0
            d_exemplar = np.zeros(len(member_indices))
            d_centroid = np.zeros(len(member_indices))
            outlier_threshold = 1.0
            split_signal = False

        core_face_ids = set(faces[i].face_id for i in (result.summary.get("core_face_ids") or []))
        face_rows = []
        for j, i in enumerate(member_indices):
            face = faces[i]
            is_core = face.is_core
            fr = _face_row(
                face=face,
                cluster_id=cluster_id,
                dist_to_exemplar=float(d_exemplar[j]),
                dist_to_centroid=float(d_centroid[j]),
                exemplar_ids=ex_face_ids,
                outlier_threshold=outlier_threshold,
                is_core=is_core,
            )
            face_rows.append(fr)
        face_rows.sort(key=lambda r: r.dist_to_exemplar)

        outlier_face_ids = [r.face_id for r in face_rows if r.is_outlier]

        # Merge threshold from run config (not hardcoded)
        run_cfg = result.summary.get("config") or {}
        merge_threshold = float(run_cfg.get("merge_candidate_threshold", 0.45))

        # Nearest clusters
        my_ex_indices = cr.exemplars.get(cluster_id, member_indices[:1])
        my_ex_mat = _embeddings_matrix(faces, my_ex_indices)

        nearest: List[NearestClusterRow] = []
        for other_cid, other_members in cr.clusters.items():
            if other_cid == cluster_id:
                continue
            other_ex_indices = cr.exemplars.get(other_cid, other_members[:1])
            other_ex_mat = _embeddings_matrix(faces, other_ex_indices)
            other_mat = _embeddings_matrix(faces, other_members)

            if my_ex_mat is None or other_ex_mat is None:
                continue

            # Min exemplar-to-exemplar distance
            cross_sims = my_ex_mat @ other_ex_mat.T
            min_ex_dist = float(np.clip(1.0 - cross_sims.max(), 0.0, 2.0))

            # p10 cross distance (all members vs all members)
            p10_dist = min_ex_dist
            if other_mat is not None and mat is not None:
                cross_all = mat @ other_mat.T
                all_dists = np.clip(1.0 - cross_all, 0.0, 2.0).flatten()
                p10_dist = float(np.percentile(all_dists, 10))

            nearest.append(NearestClusterRow(
                cluster_id=other_cid,
                size=len(other_members),
                min_exemplar_dist=round(min_ex_dist, 4),
                p10_cross_dist=round(p10_dist, 4),
                merge_threshold=merge_threshold,
                merge_candidate=min_ex_dist < merge_threshold,
            ))

        nearest.sort(key=lambda r: r.min_exemplar_dist)

        return cls(
            cluster_id=cluster_id,
            size=len(member_indices),
            diameter=round(diameter, 4),
            avg_intra_dist=round(avg_intra, 4),
            exemplar_face_ids=ex_face_ids,
            faces=face_rows,
            outlier_face_ids=outlier_face_ids,
            split_signal=split_signal,
            nearest_clusters=nearest[:10],
        )
