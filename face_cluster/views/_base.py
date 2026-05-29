"""Shared row types and low-level compute helpers for face cluster views."""
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
    cluster_id: int
    size: int
    diameter: float          # max intra-cluster distance
    avg_intra_dist: float    # mean intra-cluster distance
    n_exemplars: int
    nearest_cluster_id: int
    nearest_cluster_dist: float
    merge_candidate: bool    # exemplar dist < distance_threshold


@dataclass
class NearestClusterRow:
    cluster_id: int
    size: int
    min_exemplar_dist: float
    p10_cross_dist: float
    merge_threshold: float
    merge_candidate: bool


@dataclass
class FaceRow:
    face_id: int
    image_path: str
    blur_score: float
    area: float
    pose: Optional[Tuple[float, float, float]]  # (yaw, pitch, roll)
    dist_to_exemplar: float
    dist_to_centroid: float
    role: str   # "exemplar" | "core" | "attached" | "holdout"
    cluster_id: int
    is_outlier: bool


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
    face_id: int
    image_path: str
    cluster_id: int
    distance: float


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


def _face_row(
    face: FaceRecord,
    cluster_id: int,
    dist_to_exemplar: float,
    dist_to_centroid: float,
    exemplar_ids: List[int],
    outlier_threshold: float,
    is_core: bool,
) -> FaceRow:
    if face.face_id in exemplar_ids:
        role = "exemplar"
    elif is_core:
        role = "core"
    else:
        role = "attached"
    return FaceRow(
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
    )
