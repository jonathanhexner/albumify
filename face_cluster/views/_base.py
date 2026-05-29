"""Shared row types and low-level compute helpers for face cluster views.

These dataclasses are the typed shapes returned by the read-side view
modules (``cluster_view``, ``cluster_debug_view``, ``run_overview``,
``face_view``, ``merge_view``) and the spec-045
``ClusterAnalysisService`` / ``ClusterAnalysisRepository``. Keeping them
in one module means every consumer of "what does a cluster row look like"
agrees on the field set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_cluster.types import FaceRecord


# ---------------------------------------------------------------------------
# Small row types used inside views
# ---------------------------------------------------------------------------

@dataclass
class ClusterRow:
    """One cluster's summary row in the Cluster Analysis tab.

    Built per cluster by ``ClusterAnalysisRepository.get_cluster_rows()``;
    the compute-side fields (``nearest_cluster_*``, ``merge_candidate``)
    are placeholders at Repository level and filled in by the Service /
    ``ClusterView.compute`` when a cluster is selected.
    """
    cluster_id: int                   # cluster id from cluster_assignments; never NOISE_LABEL in this row
    size: int                         # number of member faces (core + attached)
    diameter: float                   # max pairwise cosine distance among members
    avg_intra_dist: float             # mean pairwise cosine distance among members
    n_exemplars: int                  # count of faces with is_exemplar=1
    nearest_cluster_id: int           # cluster id of the nearest neighbor (NOISE_LABEL = unset)
    nearest_cluster_dist: float       # min exemplar-to-exemplar distance to that cluster
    merge_candidate: bool             # nearest_cluster_dist < distance_threshold


@dataclass
class NearestClusterRow:
    """One row in a cluster's "nearest clusters" list (rendered by the
    nearest_clusters component). Always references some OTHER cluster
    relative to a focus cluster the user has selected.
    """
    cluster_id: int                   # the OTHER cluster's id
    size: int                         # size of the OTHER cluster
    min_exemplar_dist: float          # min cosine distance: any exemplar of focus → any exemplar of OTHER
    p10_cross_dist: float             # 10th-percentile of all cross-cluster pair distances
    merge_threshold: float            # the FCConfig merge_candidate_threshold at write time
    merge_candidate: bool             # min_exemplar_dist < merge_threshold


@dataclass
class FaceRow:
    """One face's display row in the Cluster Analysis face grid.

    Built per cluster by ``ClusterView.compute`` (and ``face_view`` for the
    per-face popup). Sorted by ``dist_to_exemplar`` ascending in the grid.
    """
    face_id: int                                  # stable id from the producer pipeline (DB primary key)
    image_path: str                               # source photo path; "" if unknown
    blur_score: float                             # quality score (higher = sharper); 0.0 if unmeasured
    area: float                                   # raw bbox area (units vary by source — see SIGHTING-060)
    pose: Optional[Tuple[float, float, float]]   # (yaw, pitch, roll) degrees; None if pose extractor didn't run
    dist_to_exemplar: float                       # cosine distance to nearest exemplar of this cluster (sort key)
    dist_to_centroid: float                       # cosine distance to this cluster's mean vector
    role: str                                     # "exemplar" | "core" | "attached" | "holdout"
    cluster_id: int                               # which cluster owns this row (denormalized)
    is_outlier: bool                              # dist_to_exemplar > p90 of cluster distances
    # Bbox area as a fraction of image area, in [0, 1]. Populated by spec-040
    # v5 producers (SIGHTING-064 fix); None for legacy v4 runs that didn't
    # write the area_ratio column. Display as percent: f"{x:.1%}".
    area_ratio: Optional[float] = None

    @classmethod
    def from_face(
        cls,
        face: FaceRecord,
        *,
        cluster_id: int,
        dist_to_exemplar: float,
        dist_to_centroid: float,
        exemplar_ids: List[int],
        outlier_threshold: float,
        is_core: bool,
    ) -> "FaceRow":
        """Build a FaceRow from a FaceRecord + per-cluster compute context.

        Replaces the legacy 7-param ``_face_row`` free function (refactored
        2026-05-29 per spec-045 §"_face_row smell"). Keyword-only args make
        call sites self-documenting and prevent positional-argument drift.

        Args:
            face: the source FaceRecord (from RunStore.faces()).
            cluster_id: id of the cluster this row belongs to.
            dist_to_exemplar: cosine distance to nearest exemplar.
            dist_to_centroid: cosine distance to cluster centroid.
            exemplar_ids: face_ids of this cluster's exemplars (used to set role).
            outlier_threshold: dist_to_exemplar above this flags the face as outlier.
            is_core: True if the face passed the quality gate (was in core_indices).

        Role assignment: ``"exemplar"`` if face_id is in exemplar_ids;
        else ``"core"`` if is_core; else ``"attached"`` (holdout-then-attached).
        """
        if face.face_id in exemplar_ids:
            role = "exemplar"
        elif is_core:
            role = "core"
        else:
            role = "attached"
        return cls(
            face_id=face.face_id,
            image_path=face.image_path or "",
            blur_score=face.blur_score,
            area=face.area,
            pose=face.pose,
            dist_to_exemplar=dist_to_exemplar,
            dist_to_centroid=dist_to_centroid,
            role=role,
            cluster_id=cluster_id,
            is_outlier=dist_to_exemplar > outlier_threshold,
            area_ratio=face.area_ratio,
        )


@dataclass
class EdgeInfo:
    """One edge in the kNN graph within a cluster."""
    face_id_a: int
    face_id_b: int
    distance: float


@dataclass
class FaceGraphInfo:
    """Per-face graph connectivity within a cluster."""
    face_id: int
    n_edges: int             # direct edges to other cluster members
    neighbors: List[int]     # face_ids of direct neighbors
    neighbor_dists: List[float]
    is_bridge: bool          # articulation point — removal splits cluster


@dataclass
class CloseFace:
    """One nearby face surfaced by the Face Analysis popup.

    Returned by ``face_view`` when the user inspects a single face and
    asks "who looks most like this one across the run?" Distance is
    cosine distance from the focus face's embedding.
    """
    face_id: int                      # the OTHER face's id
    image_path: str                   # source photo path
    cluster_id: int                   # which cluster the OTHER face is in
    distance: float                   # cosine distance from focus face to this one


@dataclass(frozen=True, slots=True)
class Assignment:
    """One row of cluster_assignments — face → cluster + exemplar flag.

    spec-045 §5.5: shared typed row returned by ClusterAnalysisRepository
    (and reused by future tabs that query cluster_assignments).
    Uses NOISE_LABEL (sim_bench.pipeline.clustering_labels) for noise faces;
    never a bare -1.
    """
    face_id: int
    cluster_id: int
    is_exemplar: bool
    iteration: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embeddings_matrix(faces: List[FaceRecord], indices: List[int]) -> Optional[np.ndarray]:
    """Stack L2-normalised embeddings for the given face indices. Returns None if any missing."""
    vecs = []
    for i in indices:
        emb = faces[i].embedding_normalized if faces[i].embedding_normalized is not None else faces[i].embedding
        if emb is None:
            return None
        vecs.append(emb / (np.linalg.norm(emb) + 1e-9))
    return np.stack(vecs).astype(np.float32)


def _pairwise_distances(mat: np.ndarray) -> np.ndarray:
    """Cosine distances from L2-normalised embeddings (1 - dot product)."""
    sims = mat @ mat.T
    return np.clip(1.0 - sims, 0.0, 2.0)


def _centroid(mat: np.ndarray) -> np.ndarray:
    return mat.mean(axis=0)


def _dist_to_centroid(mat: np.ndarray) -> np.ndarray:
    c = _centroid(mat)
    c = c / (np.linalg.norm(c) + 1e-9)
    return np.clip(1.0 - mat @ c, 0.0, 2.0)


def _exemplar_dist(mat: np.ndarray, exemplar_local_indices: List[int]) -> np.ndarray:
    """Distance from each face to the nearest exemplar."""
    if not exemplar_local_indices:
        return np.zeros(len(mat))
    ex_mat = mat[exemplar_local_indices]
    sims = mat @ ex_mat.T
    return np.clip(1.0 - sims.max(axis=1), 0.0, 2.0)


# _face_row was a 7-param free function — replaced 2026-05-29 by
# FaceRow.from_face() classmethod above. The classmethod takes the same
# arguments keyword-only and lives next to the FaceRow definition.
