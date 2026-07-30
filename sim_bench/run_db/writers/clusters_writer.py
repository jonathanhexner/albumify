"""Per-table writer for `clusters` + `cluster_assignments` (spec-057).

Extracted from RunExporter._write_clusters_and_assignments.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Tuple

from face_cluster.types import ClusterResult, FaceRecord
from sim_bench.run_db.writers._common import maybe_float


def _build_parent_map(merge_log: List[Dict]) -> Dict[int, List[int]]:
    """Union-find over actually_merged=True rows → {final_id: [original parents]}."""
    if not merge_log:
        return {}
    parent: Dict[int, int] = {}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    originals: Dict[int, List[int]] = {}
    for entry in merge_log:
        if not entry["actually_merged"]:
            continue
        a, b = int(entry["cluster_a"]), int(entry["cluster_b"])
        originals.setdefault(a, [a])
        originals.setdefault(b, [b])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
            originals[ra] = originals.get(ra, [ra]) + originals.get(rb, [rb])

    out: Dict[int, List[int]] = {}
    for cid in originals:
        root = _find(cid)
        members = originals.get(root, [])
        if len(members) > 1:
            out[root] = sorted(set(members) - {root})
    return out


def write_clusters(
    conn: sqlite3.Connection,
    base_cr: ClusterResult,
    merged_cr: ClusterResult,
    merge_log: List[Dict],
    core_indices: List[int],
    faces: List[FaceRecord],
) -> None:
    cluster_rows: List[Tuple] = []
    assign_rows: List[Tuple] = []

    # Iteration 0 — base clustering result.
    for cid, members in base_cr.clusters.items():
        stats = base_cr.cluster_stats.get(cid, {})
        cluster_rows.append((
            cid, 0, len(members),
            maybe_float(stats.get("diameter")),
            maybe_float(stats.get("mean_dist")),
            "base", "[]",
        ))
        exemplars = set(base_cr.exemplars.get(cid, []))
        for node_idx in members:
            face_idx = (
                core_indices[node_idx]
                if core_indices and node_idx < len(core_indices)
                else node_idx
            )
            if 0 <= face_idx < len(faces):
                assign_rows.append((
                    faces[face_idx].face_id,
                    cid,
                    0,
                    1 if node_idx in exemplars else 0,
                    maybe_float(getattr(faces[face_idx], "d10_score", None)),
                ))

    # Final iteration — only when at least one merge actually executed.
    max_iter = max((int(e["iteration"]) for e in merge_log), default=0)
    any_merged = any(e["actually_merged"] for e in merge_log)
    if any_merged and merged_cr is not base_cr:
        parent_map = _build_parent_map(merge_log)
        for cid, members in merged_cr.clusters.items():
            stats = merged_cr.cluster_stats.get(cid, {})
            parents = parent_map.get(cid, [])
            cluster_rows.append((
                cid, max_iter, len(members),
                maybe_float(stats.get("diameter")),
                maybe_float(stats.get("mean_dist")),
                "auto_merge" if parents else "base",
                json.dumps(parents),
            ))
            exemplars = set(merged_cr.exemplars.get(cid, []))
            for node_idx in members:
                face_idx = (
                    core_indices[node_idx]
                    if core_indices and node_idx < len(core_indices)
                    else node_idx
                )
                if 0 <= face_idx < len(faces):
                    assign_rows.append((
                        faces[face_idx].face_id,
                        cid,
                        max_iter,
                        1 if node_idx in exemplars else 0,
                        None,
                    ))

    conn.executemany(
        "INSERT INTO clusters VALUES (?,?,?,?,?,?,?)",
        cluster_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO cluster_assignments VALUES (?,?,?,?,?)",
        assign_rows,
    )
