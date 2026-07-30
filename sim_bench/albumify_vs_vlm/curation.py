"""Spec-102 — pure curation helpers shared by the Albumify arm and tests.

`select_best` has no fixed-K knob (it yields ~1-2 per scene cluster, variable total), but the
experiment requires both arms to output exactly K (fairness, CV review must-fix #3). These two
pure functions turn Albumify's (scores, scene-clusters) into an ordered K-sequence:

- `select_top_k_coverage_first` — round-robin one-best-per-cluster before any cluster's second
  pick, so trimming to K does not silently drop a whole scene. Noise images (cluster -1) are each
  treated as their own singleton scene so diverse one-offs aren't starved.
- `chronological_order` — final album order by the `YYYYMMDD_HHMMSS` filename timestamp.

No pipeline / I/O dependencies, so notebooks and unit tests use them directly.
"""

from __future__ import annotations

import re
from collections import defaultdict

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def _cluster_key(stem: str, labels: dict[str, int]):
    """Scene id for a stem; each noise image (-1) becomes its own singleton scene."""
    label = labels.get(stem, -1)
    return ("noise", stem) if label == -1 else ("scene", label)


def select_top_k_coverage_first(
    candidates: list[str],
    scores: dict[str, float],
    labels: dict[str, int],
    k: int,
) -> list[str]:
    """Pick up to k stems, one-best-per-scene before any second pick (coverage first).

    Clusters are visited in descending order of their best in-cluster score; within a cluster,
    higher-scored stems are taken first. Deterministic (ties broken by stem).
    """
    if k <= 0 or not candidates:
        return []

    by_cluster: dict[object, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for stem in candidates:
        if stem in seen:
            continue
        seen.add(stem)
        by_cluster[_cluster_key(stem, labels)].append(stem)

    # Sort each cluster's members by score desc (then stem for determinism).
    for key in by_cluster:
        by_cluster[key].sort(key=lambda s: (-scores.get(s, 0.0), s))

    # Visit clusters by their best member's score desc (then key for determinism).
    ordered_clusters = sorted(
        by_cluster.values(),
        key=lambda members: (-scores.get(members[0], 0.0), members[0]),
    )

    chosen: list[str] = []
    round_idx = 0
    max_depth = max(len(m) for m in ordered_clusters)
    while len(chosen) < k and round_idx < max_depth:
        for members in ordered_clusters:
            if round_idx < len(members):
                chosen.append(members[round_idx])
                if len(chosen) == k:
                    return chosen
        round_idx += 1
    return chosen


def _ts_key(stem: str) -> tuple:
    m = _TS_RE.search(stem)
    if m:
        return (0, m.group(1) + m.group(2), stem)   # timestamped first, chronological
    return (1, "", stem)                              # untimestamped last, lexical


def chronological_order(stems: list[str]) -> list[str]:
    """Order stems by filename timestamp (album reads front-to-back in time)."""
    return sorted(stems, key=_ts_key)
