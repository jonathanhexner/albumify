"""Checkpoint #1: verify_face_clusters (spec-111 P1, subsumes spec-105).

Catches the two face-clustering error types the deterministic clusterer makes at the borderline:
  * over-split  → MERGE   (one person in two clusters; centroid distance just above the merge threshold)
  * over-merge  → SPLIT   (two people in one cluster; high internal spread)

Pure candidate-detection + a verify() that calls an INJECTED adjudicator (the VLM in production, a fake
in tests), grounded in the embedding distances. Returns a VerifyResult of MERGE/SPLIT corrections.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

from sim_bench.verification.escalation import borderline
from sim_bench.verification.types import Correction, VerifyResult, MERGE, SPLIT

__all__ = ["cosine_distance", "merge_candidates", "split_candidates", "verify_face_clusters"]

Vec = Sequence[float]


def cosine_distance(a: Vec, b: Vec) -> float:
    """1 - cosine similarity. 0 = identical direction."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return 1.0 - dot / (na * nb)


def merge_candidates(centroids: Dict[int, Vec], merge_threshold: float, band: float
                     ) -> List[Tuple[int, int]]:
    """Cluster pairs that *just failed to merge* — distance in (threshold, threshold+band].

    These are the over-split suspects: close enough to be the same person, but clustering kept them
    apart. Confident-apart pairs (distance >> threshold) are not escalated.
    """
    ids = sorted(centroids)
    pairs = [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    # margin = distance - threshold; borderline() keeps |margin| <= band, then we take the above side.
    near = borderline(pairs, margin=lambda p: cosine_distance(centroids[p[0]], centroids[p[1]]) - merge_threshold,
                      band=band)
    return [p for p in near if cosine_distance(centroids[p[0]], centroids[p[1]]) >= merge_threshold]


def split_candidates(members: Dict[int, List[Vec]], spread_threshold: float) -> List[int]:
    """Clusters whose internal spread (mean distance to centroid) exceeds the threshold — likely bimodal."""
    out = []
    for cid, embs in members.items():
        if len(embs) < 2:
            continue
        dim = len(embs[0])
        centroid = [sum(e[k] for e in embs) / len(embs) for k in range(dim)]
        spread = sum(cosine_distance(e, centroid) for e in embs) / len(embs)
        if spread >= spread_threshold:
            out.append(cid)
    return out


def verify_face_clusters(
    centroids: Dict[int, Vec],
    members: Dict[int, List[Vec]],
    adjudicate_merge: Callable[[int, int], bool],
    adjudicate_split: Callable[[int], bool],
    *,
    merge_threshold: float = 0.35,
    merge_band: float = 0.1,
    split_spread: float = 0.45,
) -> VerifyResult:
    """Review borderline clusters; adjudicate each with the injected VLM; emit MERGE/SPLIT corrections."""
    m_cands = merge_candidates(centroids, merge_threshold, merge_band)
    s_cands = split_candidates(members, split_spread)
    escalated = [f"{i}+{j}" for i, j in m_cands] + [f"split:{c}" for c in s_cands]
    corrections: List[Correction] = []

    for i, j in m_cands:
        if adjudicate_merge(i, j):
            corrections.append(Correction(MERGE, [str(i), str(j)],
                                          reason="same person — merged (borderline distance, VLM confirmed)",
                                          detail={"clusters": [i, j]}))
    for c in s_cands:
        if adjudicate_split(c):
            corrections.append(Correction(SPLIT, [str(c)],
                                          reason="two people in one cluster — split (high spread, VLM confirmed)",
                                          detail={"cluster": c}))

    return VerifyResult(checkpoint="verify_face_clusters",
                        n_reviewed=len(centroids), escalated=escalated, corrections=corrections)
