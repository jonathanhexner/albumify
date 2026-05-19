"""Dataclasses for face clustering pipeline.

spec-033 P-C C-2: ``FaceRecord`` is a Pydantic v2 BaseModel with
``extra='forbid'``. Typo'd field names at construction raise
``ValidationError`` instead of silently dropping data — closing the
"5 fields dropped at the bridge" class of bug that motivated SIGHTING-059.
Other types in this module remain dataclasses (no boundary contracts to
enforce on them).
"""

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np
import networkx as nx
from pydantic import BaseModel, ConfigDict, field_validator


@dataclass
class GateResult:
    """Result of a single quality gate evaluation."""
    value: float
    threshold: float
    passed: bool


@dataclass
class QualityVerdict:
    """Per-face quality gate verdicts from the quality gating stage.

    gates maps gate name -> GateResult.
    Gate names: "blur", "pose_yaw", "pose_pitch", "area", "top_k_per_image".
    """
    gates: Dict[str, GateResult]
    rejection_reason: Optional[str] = None  # first failing gate name; None for core faces

    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates.values())


class ClusterOrigin(str, Enum):
    BASE         = "base"
    AUTO_MERGE   = "auto_merge"
    MANUAL_MERGE = "manual_merge"
    REMERGE      = "remerge"
    UNKNOWN      = "unknown"


@dataclass
class ClusterMetadata:
    """Provenance metadata for a single cluster."""
    origin: ClusterOrigin
    parent_cluster_ids: List[int] = field(default_factory=list)


class FaceRecord(BaseModel):
    """Single face record with metadata and embeddings.

    spec-033 P-C C-2: Pydantic v2 BaseModel. ``extra='forbid'`` —
    constructing with an unknown field raises ValidationError. The pose
    validator rejects partial tuples; everything else accepts the legacy
    sentinel values (zeros, None) so existing loaders keep working.

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
        area: Face area (UNIT varies by source — see SIGHTING-060;
              FC App standalone produces px², bridge produces fraction²)
        is_core: Whether this face passed quality gating for core set
        image_path: Full path to source image (optional, for traceability)
        face_index: Index of face within source image (optional, for traceability)
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

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
    # Observability fields (spec 012) — all default None for backward compat
    quality_verdict: Optional[QualityVerdict] = None
    rejection_reason: Optional[str] = None
    det_score: Optional[float] = None
    d10_score: Optional[float] = None
    # spec-040 Phase 4 (schema v5) canonical unit-normalized geometry — SIGHTING-064.
    # All ratios in [0, 1]; image_*_px are the raw image dims so ratio = px / image_*_px
    # is reproducible at read time. All optional for backward compat with legacy
    # producers that don't populate them.
    area_ratio: Optional[float] = None
    bbox_x_ratio: Optional[float] = None
    bbox_y_ratio: Optional[float] = None
    bbox_w_ratio: Optional[float] = None
    bbox_h_ratio: Optional[float] = None
    image_width_px: Optional[int] = None
    image_height_px: Optional[int] = None

    @field_validator("pose")
    @classmethod
    def _validate_pose_is_3_tuple(cls, v):
        """If pose is provided, all 3 dims must be present.

        Prevents the SIGHTING-059-class bug where pose is partially populated
        and downstream filters silently treat missing dims as zero.
        """
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError(
                f"pose must be a 3-tuple of (yaw, pitch, roll), got length {len(v)}"
            )
        return v


@dataclass
class MergeDecisionRow:
    """One row of the merge log: evidence + verdict for a single (cluster_a, cluster_b)
    pair evaluated in a single iteration of ConservativeMerger.

    This is the contract between the merger (which produces these as dicts), the writer
    (which persists them), and the reader / UI (which renders them). Every field in the
    in-memory dict has a typed field here; nothing is dropped on the way to disk.

    Field count must equal the keys produced by `merge.py:_select_best_merge` plus
    `actually_merged` (added after winner selection). spec-030 / SIGHTING-058.
    """
    # Identity (5)
    iteration: int
    cluster_a: int
    cluster_b: int
    cluster_a_size: int
    cluster_b_size: int

    # Distance / threshold evidence (7)
    exemplar_dist: float
    threshold_used: float
    T_a: Optional[float]
    T_b: Optional[float]
    T_global: Optional[float]
    p25_cross_dist: Optional[float]
    passes_cross: Optional[bool]

    # Support gate (3)
    support: int
    unique_support: Optional[int]
    required_support: int

    # Diameter gate (2)
    post_diameter: float
    max_allowed_diameter: float

    # Margin gate (4)
    margin_gap: float                  # may be float('inf') when gate disabled
    margin_dist_to_b: float
    margin_competitor_dist: float
    margin_competitor_id: int          # -1 when no competitor

    # Per-gate verdicts (4)
    passes_exemplar: bool
    passes_support: bool
    passes_margin: bool
    passes_diameter: bool

    # Outcome (3)
    action: str                        # "merged" | "passed" | "rejected"
    actually_merged: bool              # True only for the iteration's executed winner
    rejection_reason: Optional[str]

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        """Canonical ordered field name tuple. Used by writer/reader for column ordering
        and by strict-write to validate input dicts contain exactly these keys."""
        return tuple(f.name for f in fields(cls))


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
