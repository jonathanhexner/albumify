"""Absolute diameter cap step — runs after merge as a safety net.

Spec: specs/031-max-diameter-step/spec.md
Trigger: SIGHTING-059 (cluster 4 chained 28 faces into one blob with internal
max-pairwise distance 0.867 despite every individual merge passing gate D).

The cap inspects every MERGED cluster (not base-only) and checks two thresholds:
    max_full_diameter      — max_{i,j in cluster} cosine_dist(emb_i, emb_j)
                             (sensitive to single-outlier faces, matches gate D)
    max_exemplar_diameter  — same metric restricted to the cluster's exemplars
                             (robust to lone outliers; flags systematic multi-
                              identity blobs where even the representatives
                              are far apart)

Both must pass. On violation, action="split": every face in the violating
cluster is reverted to its pre-merge `base_cluster_id` (or to noise if its
base assignment was noise). Pre-merge provenance comes from the `base_result`
ClusterResult passed in by the pipeline (the labels array before merge ran).

The step emits one `ClusterCapDecision` per merged cluster (kept or split) so
the FC App can show a panel of "what the cap did this run".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_cluster.types import ClusterResult, FaceRecord

logger = logging.getLogger(__name__)


@dataclass
class ClusterCapDecision:
    """Audit record for one cluster the cap step inspected."""
    cluster_id: int
    n_faces: int
    full_diameter: float
    exemplar_diameter: Optional[float]   # None if cluster had no exemplars
    max_full_threshold: float
    max_exemplar_threshold: float
    full_pass: bool
    exemplar_pass: bool                  # True if no exemplars (vacuous)
    action: str                          # "kept" | "split"
    reason: str
    pre_merge_components: List[int] = field(default_factory=list)
    pre_merge_sizes:      List[int] = field(default_factory=list)


@dataclass
class CapResult:
    """Output of the cap step."""
    cluster_result: ClusterResult        # potentially mutated (post-split)
    decisions:      List[ClusterCapDecision]

    @property
    def n_split(self) -> int:
        return sum(1 for d in self.decisions if d.action == "split")

    @property
    def n_kept(self) -> int:
        return sum(1 for d in self.decisions if d.action == "kept")


# -----------------------------------------------------------------------------
# Diameter computation
# -----------------------------------------------------------------------------
def _max_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Worst-case cosine distance across all pairs.

    Args:
        embeddings: (N, D) array of L2-normalized embeddings.

    Returns:
        max over all i<j of (1 - emb_i · emb_j); 0.0 if N < 2.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    # Cosine distance = 1 - cos similarity; for L2-normalized vectors,
    # cos sim = dot product.  We want the max distance, i.e. the min dot
    # product across pairs.  Compute the full Gram matrix and ignore the
    # diagonal.
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, 1.0)  # diagonal would otherwise show 1.0; force min ignores it
    min_sim = sim.min()
    return float(1.0 - min_sim)


# -----------------------------------------------------------------------------
# Step entry point
# -----------------------------------------------------------------------------
def apply_diameter_cap(
    merged_result:           ClusterResult,
    base_result:             ClusterResult,
    faces:                   List[FaceRecord],
    *,
    max_full_diameter:       float,
    max_exemplar_diameter:   float,
) -> CapResult:
    """Reject merged clusters whose diameter exceeds either threshold.

    Args:
        merged_result: ClusterResult AFTER merge has run.
        base_result:   ClusterResult BEFORE merge (used as provenance for splits).
        faces:         All face records (must have embedding_normalized populated).
        max_full_diameter:     Absolute ceiling on max pairwise cosine distance
                               across ALL faces in a cluster.
        max_exemplar_diameter: Absolute ceiling on max pairwise cosine distance
                               across the cluster's exemplars only.

    Returns:
        CapResult with the (possibly mutated) ClusterResult and per-cluster decisions.
        Mutation policy: returns a NEW ClusterResult, never edits inputs in place.
    """
    assert max_full_diameter > 0,     "max_full_diameter must be positive"
    assert max_exemplar_diameter > 0, "max_exemplar_diameter must be positive"
    assert len(base_result.labels) == len(merged_result.labels), (
        "base_result and merged_result must cover the same face set "
        f"(base={len(base_result.labels)}, merged={len(merged_result.labels)})"
    )

    # Build a copy of labels we can mutate.
    new_labels = merged_result.labels.copy()
    decisions: List[ClusterCapDecision] = []

    # Pre-fetch normalized embeddings as a single matrix for fast slicing.
    embeddings = np.stack([f.embedding_normalized for f in faces])

    for cid, face_indices in merged_result.clusters.items():
        if cid < 0 or len(face_indices) < 2:
            # Noise clusters or singletons can't violate any diameter.
            continue

        cluster_emb = embeddings[face_indices]
        full_diam   = _max_pairwise_cosine(cluster_emb)

        exemplar_indices = merged_result.exemplars.get(cid, [])
        if len(exemplar_indices) >= 2:
            exemplar_emb  = embeddings[exemplar_indices]
            exemplar_diam = _max_pairwise_cosine(exemplar_emb)
            exemplar_pass = exemplar_diam <= max_exemplar_diameter
        else:
            # No exemplars (or only one) → exemplar gate is vacuous.
            exemplar_diam = None
            exemplar_pass = True

        full_pass = full_diam <= max_full_diameter

        if full_pass and exemplar_pass:
            decisions.append(ClusterCapDecision(
                cluster_id=cid, n_faces=len(face_indices),
                full_diameter=full_diam, exemplar_diameter=exemplar_diam,
                max_full_threshold=max_full_diameter,
                max_exemplar_threshold=max_exemplar_diameter,
                full_pass=True, exemplar_pass=True,
                action="kept",
                reason=f"full={full_diam:.3f}<={max_full_diameter}, "
                       f"exemplar={exemplar_diam if exemplar_diam is None else f'{exemplar_diam:.3f}'}"
                       f"<={max_exemplar_diameter}",
            ))
            continue

        # Violation -- revert each face in this cluster to its base cluster_id.
        reason_parts: List[str] = []
        if not full_pass:
            reason_parts.append(f"full {full_diam:.3f} > {max_full_diameter}")
        if not exemplar_pass:
            reason_parts.append(
                f"exemplar {exemplar_diam:.3f} > {max_exemplar_diameter}"
            )

        component_sizes: Dict[int, int] = {}
        for face_idx in face_indices:
            base_cid = int(base_result.labels[face_idx])
            new_labels[face_idx] = base_cid
            component_sizes[base_cid] = component_sizes.get(base_cid, 0) + 1

        decisions.append(ClusterCapDecision(
            cluster_id=cid, n_faces=len(face_indices),
            full_diameter=full_diam, exemplar_diameter=exemplar_diam,
            max_full_threshold=max_full_diameter,
            max_exemplar_threshold=max_exemplar_diameter,
            full_pass=full_pass, exemplar_pass=exemplar_pass,
            action="split",
            reason="; ".join(reason_parts),
            pre_merge_components=sorted(component_sizes.keys()),
            pre_merge_sizes=[component_sizes[k] for k in sorted(component_sizes.keys())],
        ))
        logger.info(
            "Cap split cluster %d (n=%d): %s -> %d components %s",
            cid, len(face_indices), "; ".join(reason_parts),
            len(component_sizes),
            list(zip(sorted(component_sizes.keys()),
                     [component_sizes[k] for k in sorted(component_sizes.keys())])),
        )

    return CapResult(
        cluster_result=_rebuild_cluster_result(new_labels, faces, merged_result),
        decisions=decisions,
    )


# -----------------------------------------------------------------------------
# Rebuilding ClusterResult after label mutation
# -----------------------------------------------------------------------------
def _rebuild_cluster_result(
    new_labels:   np.ndarray,
    faces:        List[FaceRecord],
    template:     ClusterResult,
) -> ClusterResult:
    """Recompute clusters/stats/exemplars from mutated labels.

    Preserves exemplar selections from the template when the cluster survived
    (same cluster_id present), drops them otherwise.  cluster_stats carried
    forward unchanged for kept clusters; reverted-to clusters fall back to
    cluster_stats from the base run if available, else minimal stats.
    """
    clusters: Dict[int, List[int]]  = {}
    for face_idx, cid in enumerate(new_labels):
        cid_i = int(cid)
        clusters.setdefault(cid_i, []).append(face_idx)

    exemplars: Dict[int, List[int]] = {}
    cluster_stats: Dict[int, Dict[str, float]] = {}
    for cid in clusters:
        if cid < 0:
            continue
        # Keep template exemplars only if the cluster ID survived unchanged.
        if cid in template.clusters and clusters[cid] == template.clusters[cid]:
            exemplars[cid]     = template.exemplars.get(cid, [])
            cluster_stats[cid] = template.cluster_stats.get(cid, {})
        else:
            # Reverted cluster -- exemplars from the merged result no longer
            # correspond to this cluster's members.  Leave empty for now.
            exemplars[cid]     = []
            cluster_stats[cid] = {"size": len(clusters[cid]), "origin": "cap_split"}

    n_clusters = sum(1 for cid in clusters if cid >= 0)
    n_noise    = len(clusters.get(-1, []))
    return ClusterResult(
        labels=new_labels,
        clusters=clusters,
        cluster_stats=cluster_stats,
        exemplars=exemplars,
        n_clusters=n_clusters,
        n_noise=n_noise,
    )


# -----------------------------------------------------------------------------
# Serialization for the audit log
# -----------------------------------------------------------------------------
def decisions_to_dict_list(decisions: List[ClusterCapDecision]) -> List[Dict]:
    """Flatten ClusterCapDecisions to JSON-serializable dicts."""
    out = []
    for d in decisions:
        out.append({
            "cluster_id":              d.cluster_id,
            "n_faces":                 d.n_faces,
            "full_diameter":           d.full_diameter,
            "exemplar_diameter":       d.exemplar_diameter,
            "max_full_threshold":      d.max_full_threshold,
            "max_exemplar_threshold":  d.max_exemplar_threshold,
            "full_pass":               d.full_pass,
            "exemplar_pass":           d.exemplar_pass,
            "action":                  d.action,
            "reason":                  d.reason,
            "pre_merge_components":    d.pre_merge_components,
            "pre_merge_sizes":         d.pre_merge_sizes,
        })
    return out
