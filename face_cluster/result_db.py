"""Face clustering results DB — SQLite per-run storage with full traceability.

Replaces CSV/JSON/numpy flat files with a queryable relational DB.
One DB file per run: {output_dir}/face_clustering.db

Tables:
    faces            — immutable detection data, one row per face
    embeddings       — 512-dim vector per face (binary blob)
    cluster_assignments — face → cluster at each iteration (core traceability)
    clusters         — per-cluster metrics at each iteration
    merge_decisions  — full evidence per candidate pair per iteration
    face_scores      — pose/eyes/expression per face
    run_metadata     — config and summary
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from face_cluster.types import FaceRecord, ClusterResult

logger = logging.getLogger(__name__)

DB_FILENAME = "face_clustering.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id       INTEGER PRIMARY KEY,
    image_path    TEXT NOT NULL,
    image_id      TEXT,
    face_index    INTEGER,
    bbox_x        REAL,
    bbox_y        REAL,
    bbox_w        REAL,
    bbox_h        REAL,
    crop_path     TEXT,
    det_score     REAL,
    blur_score    REAL,
    area          REAL,
    yaw           REAL,
    pitch         REAL,
    roll          REAL,
    is_core       INTEGER,
    rejection_reason TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    face_id       INTEGER PRIMARY KEY REFERENCES faces(face_id),
    embedding     BLOB,
    model_name    TEXT,
    l2_norm       REAL
);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    face_id       INTEGER REFERENCES faces(face_id),
    cluster_id    INTEGER,
    iteration     INTEGER,
    is_exemplar   INTEGER DEFAULT 0,
    d10_score     REAL,
    PRIMARY KEY (face_id, iteration)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id    INTEGER,
    iteration     INTEGER,
    size          INTEGER,
    diameter      REAL,
    avg_intra_dist REAL,
    origin        TEXT,
    parent_ids    TEXT,
    PRIMARY KEY (cluster_id, iteration)
);

CREATE TABLE IF NOT EXISTS merge_decisions (
    iteration     INTEGER,
    cluster_a     INTEGER,
    cluster_b     INTEGER,
    action        TEXT,
    exemplar_dist REAL,
    cross_dist    REAL,
    support       INTEGER,
    margin_gap    REAL,
    post_diameter REAL,
    passes_exemplar INTEGER,
    passes_cross  INTEGER,
    passes_support INTEGER,
    passes_margin INTEGER,
    passes_diameter INTEGER,
    threshold_used REAL,
    rejection_reason TEXT,
    PRIMARY KEY (iteration, cluster_a, cluster_b)
);

CREATE TABLE IF NOT EXISTS face_scores (
    face_id       INTEGER PRIMARY KEY REFERENCES faces(face_id),
    pose_score    REAL,
    eyes_score    REAL,
    expression_score REAL,
    frontal_score REAL,
    is_clusterable INTEGER
);

CREATE TABLE IF NOT EXISTS run_metadata (
    run_id        TEXT PRIMARY KEY,
    source_album  TEXT,
    config        TEXT,
    n_images      INTEGER,
    n_faces       INTEGER,
    n_core        INTEGER,
    n_clusters_base INTEGER,
    n_clusters_final INTEGER,
    n_merges      INTEGER,
    n_iterations  INTEGER,
    started_at    TEXT,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_ca_iteration ON cluster_assignments(iteration);
CREATE INDEX IF NOT EXISTS idx_ca_cluster ON cluster_assignments(cluster_id, iteration);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_path);
CREATE INDEX IF NOT EXISTS idx_md_iteration ON merge_decisions(iteration);
"""


def write_results_db(
    faces: List[FaceRecord],
    base_cluster_result: ClusterResult,
    merged_cluster_result: ClusterResult,
    core_indices: List[int],
    merge_log: Optional[List[Dict]],
    merge_metadata: Optional[Dict],
    config,
    output_dir: Path,
    source_album: str,
    crop_manifest: Optional[Dict] = None,
) -> Path:
    """Write face clustering results to a SQLite DB.

    Args:
        faces: All FaceRecord objects (core + holdout)
        base_cluster_result: Pre-merge clustering result
        merged_cluster_result: Post-merge result (same as base if no merge)
        core_indices: Maps graph node index → face list index
        merge_log: List of merge decision dicts (from ConservativeMerger)
        merge_metadata: Merge thresholds and config
        config: PipelineConfig
        output_dir: Directory to write face_clustering.db into
        source_album: Album name/path
        crop_manifest: face_id → crop_path mapping

    Returns:
        Path to the created DB file.
    """
    db_path = output_dir / DB_FILENAME
    crop_manifest = crop_manifest or {}

    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)

    try:
        _write_faces(conn, faces, crop_manifest)
        _write_embeddings(conn, faces, config)
        _write_face_scores(conn, faces)
        _write_base_clusters(conn, base_cluster_result, core_indices, faces)
        if merge_log:
            _write_merge_iterations(conn, merge_log, base_cluster_result,
                                    merged_cluster_result, core_indices, faces)
        _write_run_metadata(conn, faces, base_cluster_result, merged_cluster_result,
                            merge_log, config, source_album)
        conn.commit()
        logger.info(f"Wrote face clustering DB to {db_path}")
    except Exception as e:
        logger.error(f"Failed to write results DB: {e}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()

    return db_path


def _write_faces(conn: sqlite3.Connection, faces: List[FaceRecord],
                  crop_manifest: Dict) -> None:
    """Write faces table."""
    rows = []
    for face in faces:
        bbox = face.bbox or (0, 0, 0, 0)
        pose = face.pose or (None, None, None)
        yaw, pitch, roll = (pose[0], pose[1], pose[2]) if len(pose) >= 3 else (None, None, None)

        rows.append((
            face.face_id,
            face.image_path or face.image_id,
            face.image_id,
            face.face_index,
            float(bbox[0]) if bbox[0] is not None else None,
            float(bbox[1]) if bbox[1] is not None else None,
            float(bbox[2]) if bbox[2] is not None else None,
            float(bbox[3]) if bbox[3] is not None else None,
            str(crop_manifest.get(face.face_id, "")),
            float(face.det_score) if face.det_score is not None else None,
            float(face.blur_score),
            float(face.area),
            float(yaw) if yaw is not None else None,
            float(pitch) if pitch is not None else None,
            float(roll) if roll is not None else None,
            1 if face.is_core else 0,
            face.rejection_reason,
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    logger.info(f"  Wrote {len(rows)} faces")


def _write_embeddings(conn: sqlite3.Connection, faces: List[FaceRecord],
                       config) -> None:
    """Write embeddings table."""
    model_name = getattr(config, "embedding_model", "buffalo_l")
    rows = []
    for face in faces:
        emb = face.embedding_normalized
        if emb is None:
            emb = face.embedding
        if emb is not None:
            emb_array = np.asarray(emb, dtype=np.float32)
            l2 = float(np.linalg.norm(emb_array))
            rows.append((
                face.face_id,
                emb_array.tobytes(),
                model_name,
                round(l2, 4),
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
        rows
    )
    logger.info(f"  Wrote {len(rows)} embeddings")


def _write_face_scores(conn: sqlite3.Connection, faces: List[FaceRecord]) -> None:
    """Write face_scores table from FaceRecord quality data."""
    rows = []
    for face in faces:
        pose = face.pose
        pose_score = None
        if pose and len(pose) >= 2:
            # Frontal score: 1.0 when yaw=0, 0.0 when yaw=90
            yaw = abs(pose[0]) if pose[0] is not None else 90
            pose_score = max(0.0, 1.0 - yaw / 90.0)

        rows.append((
            face.face_id,
            pose_score,
            None,  # eyes_score — not available on FaceRecord
            None,  # expression_score — not available on FaceRecord
            None,  # frontal_score — computed differently per pipeline
            1 if face.is_core else 0,  # is_clusterable approximation
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO face_scores VALUES (?,?,?,?,?,?)",
        rows
    )
    logger.info(f"  Wrote {len(rows)} face_scores")


def _write_base_clusters(conn: sqlite3.Connection, cluster_result: ClusterResult,
                          core_indices: List[int], faces: List[FaceRecord]) -> None:
    """Write cluster_assignments and clusters for iteration 0 (base clustering)."""
    # Map graph node indices → face list indices
    assignment_rows = []
    for cid, node_indices in cluster_result.clusters.items():
        exemplar_set = set(cluster_result.exemplars.get(cid, []))
        for node_idx in node_indices:
            face_idx = core_indices[node_idx] if node_idx < len(core_indices) else node_idx
            if face_idx < len(faces):
                fid = faces[face_idx].face_id
                is_ex = 1 if node_idx in exemplar_set else 0
                d10 = float(faces[face_idx].d10_score) if faces[face_idx].d10_score is not None else None
                assignment_rows.append((fid, cid, 0, is_ex, d10))

    conn.executemany(
        "INSERT OR REPLACE INTO cluster_assignments VALUES (?,?,?,?,?)",
        assignment_rows
    )
    logger.info(f"  Wrote {len(assignment_rows)} cluster_assignments (iteration 0)")

    # Write clusters table iteration 0
    cluster_rows = []
    for cid, members in cluster_result.clusters.items():
        stats = cluster_result.cluster_stats.get(cid, {})
        cluster_rows.append((
            cid, 0, len(members),
            _safe_float(stats.get("diameter")),
            _safe_float(stats.get("mean_dist")),
            "base", "[]",
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO clusters VALUES (?,?,?,?,?,?,?)",
        cluster_rows
    )
    logger.info(f"  Wrote {len(cluster_rows)} clusters (iteration 0)")


def _write_merge_iterations(conn: sqlite3.Connection, merge_log: List[Dict],
                             base_cr: ClusterResult, merged_cr: ClusterResult,
                             core_indices: List[int], faces: List[FaceRecord]) -> None:
    """Write merge_decisions and per-iteration cluster_assignments/clusters."""
    # Write ALL merge decisions
    decision_rows = []
    for entry in merge_log:
        decision_rows.append((
            int(entry.get("iteration", 0)),
            int(entry.get("cluster_a", 0)),
            int(entry.get("cluster_b", 0)),
            str(entry.get("action", "")),
            _safe_float(entry.get("exemplar_dist")),
            _safe_float(entry.get("p25_cross_dist")),
            _safe_int(entry.get("support")),
            _safe_float(entry.get("margin_gap")),
            _safe_float(entry.get("post_diameter")),
            _safe_bool(entry.get("passes_exemplar")),
            _safe_bool(entry.get("passes_cross")),
            _safe_bool(entry.get("passes_support")),
            _safe_bool(entry.get("passes_margin")),
            _safe_bool(entry.get("passes_diameter")),
            _safe_float(entry.get("threshold_used")),
            entry.get("rejection_reason"),
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO merge_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        decision_rows
    )
    logger.info(f"  Wrote {len(decision_rows)} merge_decisions")

    # Write final-iteration cluster_assignments from merged_cluster_result
    max_iter = max((e.get("iteration", 0) for e in merge_log), default=0)
    if any(e.get("actually_merged") for e in merge_log):
        final_rows = []
        for cid, node_indices in merged_cr.clusters.items():
            exemplar_set = set(merged_cr.exemplars.get(cid, []))
            for node_idx in node_indices:
                face_idx = core_indices[node_idx] if node_idx < len(core_indices) else node_idx
                if face_idx < len(faces):
                    fid = faces[face_idx].face_id
                    is_ex = 1 if node_idx in exemplar_set else 0
                    final_rows.append((fid, cid, max_iter, is_ex, None))

        conn.executemany(
            "INSERT OR REPLACE INTO cluster_assignments VALUES (?,?,?,?,?)",
            final_rows
        )
        logger.info(f"  Wrote {len(final_rows)} cluster_assignments (iteration {max_iter}, final)")

        # Write final clusters
        final_cluster_rows = []
        parent_map = _build_parent_map(merge_log)
        for cid, members in merged_cr.clusters.items():
            stats = merged_cr.cluster_stats.get(cid, {})
            parents = parent_map.get(cid, [])
            origin = "auto_merge" if parents else "base"
            final_cluster_rows.append((
                cid, max_iter, len(members),
                _safe_float(stats.get("diameter")),
                _safe_float(stats.get("mean_dist")),
                origin,
                json.dumps(parents),
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO clusters VALUES (?,?,?,?,?,?,?)",
            final_cluster_rows
        )
        logger.info(f"  Wrote {len(final_cluster_rows)} clusters (iteration {max_iter}, final)")


def _write_run_metadata(conn: sqlite3.Connection, faces: List[FaceRecord],
                         base_cr: ClusterResult, merged_cr: ClusterResult,
                         merge_log: Optional[List[Dict]], config,
                         source_album: str) -> None:
    """Write run_metadata table."""
    n_merges = sum(1 for e in (merge_log or []) if e.get("actually_merged"))
    n_iterations = max((e.get("iteration", 0) for e in (merge_log or [])), default=0)
    now = datetime.now().isoformat()

    try:
        config_json = json.dumps(
            {k: v for k, v in vars(config).items() if not k.startswith("_")},
            default=str
        )
    except Exception:
        config_json = "{}"

    conn.execute(
        "INSERT OR REPLACE INTO run_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            now,  # run_id
            source_album,
            config_json,
            len(set(f.image_path or f.image_id for f in faces)),  # n_images
            len(faces),
            sum(1 for f in faces if f.is_core),
            base_cr.n_clusters,
            merged_cr.n_clusters,
            n_merges,
            n_iterations,
            now,
            now,
        )
    )
    logger.info(f"  Wrote run_metadata")


def _build_parent_map(merge_log: List[Dict]) -> Dict[int, List[int]]:
    """Build map: merged_cluster_id → list of original parent cluster IDs."""
    parents = defaultdict(set)
    for entry in merge_log:
        if entry.get("actually_merged"):
            ca, cb = entry["cluster_a"], entry["cluster_b"]
            # The merged result keeps cluster_a's ID (convention)
            parents[ca].add(cb)
    return {k: sorted(v) for k, v in parents.items()}


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_bool(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return 1 if bool(v) else 0
    except (TypeError, ValueError):
        return None
