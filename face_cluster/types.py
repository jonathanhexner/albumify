"""Dataclasses for face clustering pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import networkx as nx


@dataclass
class FaceRecord:
    """Single face record with metadata and embeddings.

    Attributes:
        face_id: Unique identifier (int or str)
        image_id: Image identifier this face belongs to (typically basename)
        bbox: Bounding box (x1, y1, x2, y2)
        landmarks: Facial landmarks if available (5x2 or Nx2 array)
        aligned_face: Aligned face crop (HxWx3 uint8 array)
        embedding: Face embedding vector (typically 512-dim)
        embedding_normalized: L2-normalized embedding
        pose: (yaw, pitch, roll) in degrees, None if not computed
        blur_score: Laplacian variance blur score
        area: Face area in pixels (bbox width * height)
        is_core: Whether this face passed quality gating for core set
        image_path: Full path to source image (optional, for traceability)
        face_index: Index of face within source image (optional, for traceability)
    """
    face_id: int
    image_id: str
    bbox: Tuple[float, float, float, float]
    landmarks: Optional[np.ndarray] = None
    aligned_face: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    embedding_normalized: Optional[np.ndarray] = None
    pose: Optional[Tuple[float, float, float]] = None
    blur_score: float = 0.0
    area: float = 0.0
    is_core: bool = False
    image_path: Optional[str] = None
    face_index: Optional[int] = None


@dataclass
class GraphResult:
    """Result of mutual kNN graph construction.

    Attributes:
        neighbors: List of neighbor indices for each node
        neighbor_distances: Corresponding distances for each neighbor
        edges: List of (i, j, distance) tuples for edges that passed threshold
        G: NetworkX graph with nodes and edges
        distance_matrix: Full pairwise distance matrix (for visualization)
    """
    neighbors: List[List[int]]
    neighbor_distances: List[List[float]]
    edges: List[Tuple[int, int, float]]
    G: nx.Graph
    distance_matrix: np.ndarray


@dataclass
class ClusterResult:
    """Result of clustering operation.

    Attributes:
        labels: Cluster label per face (-1 for noise)
        clusters: Dict mapping cluster_id -> list of face indices
        cluster_stats: Dict mapping cluster_id -> statistics dict
        exemplars: Dict mapping cluster_id -> list of exemplar face indices
        n_clusters: Number of clusters (excluding noise)
        n_noise: Number of noise points
    """
    labels: np.ndarray
    clusters: Dict[int, List[int]]
    cluster_stats: Dict[int, Dict[str, float]]
    exemplars: Dict[int, List[int]] = field(default_factory=dict)
    n_clusters: int = 0
    n_noise: int = 0
