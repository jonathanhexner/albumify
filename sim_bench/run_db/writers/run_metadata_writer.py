"""Per-table writer for `run_metadata` (spec-057).

Extracted from RunExporter._write_run_metadata.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional

from face_cluster.types import ClusterResult, FaceRecord
from sim_bench.run_db._schema import SCHEMA_VERSION


def write_run_metadata(
    conn: sqlite3.Connection,
    *,
    faces: List[FaceRecord],
    base_cr: ClusterResult,
    merged_cr: ClusterResult,
    merge_log: List[Dict],
    merge_metadata: Optional[Dict],
    config,
    source_album: str,
    producer: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    parent_run_id: Optional[str],
) -> None:
    n_merges = sum(1 for e in merge_log if e["actually_merged"])
    n_iterations = max((int(e["iteration"]) for e in merge_log), default=0)

    try:
        config_json = json.dumps(
            {k: v for k, v in vars(config).items() if not k.startswith("_")},
            default=str,
        )
    except TypeError:
        config_json = "{}"

    thresholds_json = None
    iter_summary_json = None
    if merge_metadata:
        thresholds = {
            k: merge_metadata[k]
            for k in (
                "cluster_thresholds", "global_threshold",
                "merge_exemplar_threshold", "merge_candidate_threshold",
            )
            if k in merge_metadata
        }
        if thresholds:
            thresholds_json = json.dumps(thresholds, default=str)

        iter_summary = {
            k: merge_metadata[k]
            for k in ("n_iterations", "n_candidates_proposed")
            if k in merge_metadata
        }
        if iter_summary:
            iter_summary_json = json.dumps(iter_summary, default=str)

    n_images = len({f.image_path or f.image_id for f in faces})
    n_core = sum(1 for f in faces if f.is_core)

    conn.execute(
        "INSERT INTO run_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            source_album,
            producer,
            parent_run_id,
            config_json,
            thresholds_json,
            iter_summary_json,
            n_images,
            len(faces),
            n_core,
            base_cr.n_clusters,
            merged_cr.n_clusters,
            n_merges,
            n_iterations,
            started_at,
            finished_at,
            SCHEMA_VERSION,
        ),
    )
