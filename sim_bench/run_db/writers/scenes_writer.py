"""Per-table writer for `scene_clusters` + `scene_cluster_assignments` (spec-057).

Extracted from RunExporter._write_scene_clusters and
RunExporter._write_scene_cluster_assignments.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from face_cluster.db import SCENE_CLUSTERS_SCHEMA, SCENE_CLUSTER_ASSIGNMENTS_SCHEMA
from sim_bench.run_db.writers._common import maybe_float


def write_scene_clusters(
    conn: sqlite3.Connection,
    scene_clusters: Optional[List[Dict]] = None,
) -> None:
    """Persist scene clusters to the `scene_clusters` table.

    Pandera-validates even the empty DataFrame so the contract is
    exercised on every run.

    Each input dict must carry: scene_cluster_id (int), iteration (int),
    size (int), method (str). Optional: exemplar_image_path,
    avg_intra_distance, created_at.
    """
    rows: List[Tuple] = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for sc in scene_clusters or []:
        rows.append((
            int(sc["scene_cluster_id"]),
            int(sc.get("iteration", 0)),
            int(sc["size"]),
            str(sc.get("method", "")),
            sc.get("exemplar_image_path"),
            maybe_float(sc.get("avg_intra_distance")),
            sc.get("created_at", now),
        ))
    df = pd.DataFrame(rows, columns=[
        "scene_cluster_id", "iteration", "size", "method",
        "exemplar_image_path", "avg_intra_distance", "created_at",
    ])
    SCENE_CLUSTERS_SCHEMA.validate(df)
    if rows:
        conn.executemany(
            "INSERT INTO scene_clusters "
            "(scene_cluster_id, iteration, size, method, "
            " exemplar_image_path, avg_intra_distance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def write_scene_cluster_assignments(
    conn: sqlite3.Connection,
    scene_cluster_assignments: Optional[List[Dict]] = None,
) -> None:
    """Persist (image_path, scene_cluster_id, iteration) tuples.

    Each input dict must carry image_path, scene_cluster_id, iteration.
    Optional: distance_to_centroid.
    """
    rows: List[Tuple] = []
    for sca in scene_cluster_assignments or []:
        rows.append((
            str(sca["image_path"]),
            int(sca["scene_cluster_id"]),
            int(sca.get("iteration", 0)),
            maybe_float(sca.get("distance_to_centroid")),
        ))
    df = pd.DataFrame(rows, columns=[
        "image_path", "scene_cluster_id", "iteration", "distance_to_centroid",
    ])
    SCENE_CLUSTER_ASSIGNMENTS_SCHEMA.validate(df)
    if rows:
        conn.executemany(
            "INSERT INTO scene_cluster_assignments "
            "(image_path, scene_cluster_id, iteration, distance_to_centroid) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
