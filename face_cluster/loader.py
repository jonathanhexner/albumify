"""Load a completed pipeline run from its output directory.

Reads faces.csv, clusters.csv, embeddings.npy, embedding_face_ids.npy, and
pipeline_run.json to reconstruct a PipelineResult without re-running any model.

Schema contract:
    faces.csv              — face_id, image_path, image_id, crop_path, cluster_id,
                             is_core, blur_score, area, yaw, pitch, roll
    clusters.csv           — cluster_id, size, exemplar_face_ids (JSON list), diameter
    embeddings.npy         — float32 (n_faces x 512), L2-normalised, row-aligned with faces
    embedding_face_ids.npy — int32 (n_faces,), face_id per embedding row
    pipeline_run.json      — run metadata written by pipeline.py
"""
from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from face_cluster.export import load_merge_decisions
from face_cluster.pipeline import PipelineResult
from face_cluster.types import ClusterResult, FaceRecord, QualityVerdict, GateResult

logger = logging.getLogger(__name__)


def load_pipeline_result(run_dir: Path) -> PipelineResult:
    """Reconstruct a PipelineResult from a completed run directory.

    Tries SQLite DB first (face_clustering.db), falls back to CSV/numpy files.
    """
    run_dir = Path(run_dir)
    db_path = run_dir / "face_clustering.db"

    if db_path.exists():
        try:
            result = _load_from_db(run_dir, db_path)
            logger.info(f"Loaded run from DB: {db_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load from DB, falling back to CSVs: {e}")

    return _load_from_csvs(run_dir)


def _load_from_db(run_dir: Path, db_path: Path) -> PipelineResult:
    """Load PipelineResult from face_clustering.db."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load faces
    face_rows = conn.execute("SELECT * FROM faces ORDER BY face_id").fetchall()
    emb_rows = {r["face_id"]: r for r in conn.execute("SELECT * FROM embeddings").fetchall()}

    faces = []
    for row in face_rows:
        fid = row["face_id"]
        emb_data = emb_rows.get(fid)
        emb = None
        if emb_data and emb_data["embedding"]:
            emb = np.frombuffer(emb_data["embedding"], dtype=np.float32)

        pose = None
        if row["yaw"] is not None:
            pose = (row["yaw"], row["pitch"], row["roll"])

        face = FaceRecord(
            face_id=fid,
            image_id=row["image_id"] or "",
            bbox=(row["bbox_x"] or 0, row["bbox_y"] or 0, row["bbox_w"] or 0, row["bbox_h"] or 0),
            embedding_normalized=emb,
            blur_score=row["blur_score"] or 0.0,
            area=row["area"] or 0.0,
            is_core=bool(row["is_core"]),
            image_path=row["image_path"],
            face_index=row["face_index"],
            pose=pose,
            det_score=row["det_score"],
            rejection_reason=row["rejection_reason"],
        )
        faces.append(face)

    # Load base cluster assignments (iteration 0)
    base_assignments = conn.execute(
        "SELECT face_id, cluster_id, is_exemplar, d10_score FROM cluster_assignments WHERE iteration = 0"
    ).fetchall()

    face_id_to_idx = {f.face_id: i for i, f in enumerate(faces)}
    clusters = defaultdict(list)
    exemplars = defaultdict(list)
    for row in base_assignments:
        idx = face_id_to_idx.get(row["face_id"])
        if idx is not None:
            clusters[row["cluster_id"]].append(idx)
            if row["is_exemplar"]:
                exemplars[row["cluster_id"]].append(idx)
            if row["d10_score"] is not None:
                faces[idx].d10_score = row["d10_score"]

    labels = np.full(len(faces), -1, dtype=np.int32)
    for cid, indices in clusters.items():
        for idx in indices:
            labels[idx] = cid

    # Load cluster stats (iteration 0)
    cluster_stats = {}
    for row in conn.execute("SELECT * FROM clusters WHERE iteration = 0").fetchall():
        cluster_stats[row["cluster_id"]] = {
            "diameter": row["diameter"],
            "mean_dist": row["avg_intra_dist"],
            "size": row["size"],
        }

    cluster_result = ClusterResult(
        labels=labels,
        clusters=dict(clusters),
        cluster_stats=cluster_stats,
        exemplars=dict(exemplars),
        n_clusters=len(clusters),
        n_noise=int((labels == -1).sum()),
    )

    # Load merged cluster result (max iteration)
    max_iter_row = conn.execute(
        "SELECT MAX(iteration) as m FROM cluster_assignments WHERE iteration > 0"
    ).fetchone()
    max_iter = max_iter_row["m"] if max_iter_row and max_iter_row["m"] else None

    merged_cluster_result = None
    if max_iter is not None:
        merged_assignments = conn.execute(
            "SELECT face_id, cluster_id, is_exemplar FROM cluster_assignments WHERE iteration = ?",
            (max_iter,)
        ).fetchall()

        m_clusters = defaultdict(list)
        m_exemplars = defaultdict(list)
        m_labels = np.full(len(faces), -1, dtype=np.int32)
        for row in merged_assignments:
            idx = face_id_to_idx.get(row["face_id"])
            if idx is not None:
                m_clusters[row["cluster_id"]].append(idx)
                m_labels[idx] = row["cluster_id"]
                if row["is_exemplar"]:
                    m_exemplars[row["cluster_id"]].append(idx)

        m_stats = {}
        for row in conn.execute("SELECT * FROM clusters WHERE iteration = ?", (max_iter,)).fetchall():
            m_stats[row["cluster_id"]] = {
                "diameter": row["diameter"],
                "mean_dist": row["avg_intra_dist"],
                "size": row["size"],
            }

        merged_cluster_result = ClusterResult(
            labels=m_labels,
            clusters=dict(m_clusters),
            cluster_stats=m_stats,
            exemplars=dict(m_exemplars),
            n_clusters=len(m_clusters),
            n_noise=int((m_labels == -1).sum()),
        )

    # Load merge log
    merge_log = None
    merge_rows = conn.execute("SELECT * FROM merge_decisions ORDER BY iteration, cluster_a").fetchall()
    if merge_rows:
        merge_log = [dict(row) for row in merge_rows]

    # Load merge metadata from run_metadata
    merge_metadata = None
    meta_row = conn.execute("SELECT * FROM run_metadata LIMIT 1").fetchone()
    summary = {}
    if meta_row:
        summary = {
            "n_faces": meta_row["n_faces"],
            "n_core": meta_row["n_core"],
            "n_clusters": meta_row["n_clusters_base"],
            "n_clusters_merged": meta_row["n_clusters_final"],
        }
        try:
            config_data = json.loads(meta_row["config"]) if meta_row["config"] else {}
            merge_metadata = {
                "n_iterations": meta_row["n_iterations"],
                "merge_exemplar_threshold": config_data.get("merge_exemplar_threshold"),
                "merge_candidate_threshold": config_data.get("merge_candidate_threshold"),
            }
        except Exception:
            pass

    # Load merge decisions (user approvals)
    merge_decisions = None
    decisions_path = run_dir / "merge_decisions.json"
    if decisions_path.exists():
        try:
            with open(decisions_path, encoding="utf-8") as fh:
                merge_decisions = json.load(fh)
        except Exception:
            pass

    conn.close()

    return PipelineResult(
        faces=faces,
        cluster_result=cluster_result,
        output_dir=run_dir,
        summary=summary,
        merged_cluster_result=merged_cluster_result,
        merge_log=merge_log,
        merge_metadata=merge_metadata,
        merge_decisions=merge_decisions,
    )


def _load_from_csvs(run_dir: Path) -> PipelineResult:
    """Legacy loader: reconstruct PipelineResult from CSV/numpy/JSON files."""

    faces_csv = run_dir / "faces.csv"
    run_json  = run_dir / "pipeline_run.json"

    if not faces_csv.exists():
        raise FileNotFoundError(f"faces.csv not found in {run_dir}")
    if not run_json.exists():
        raise FileNotFoundError(f"pipeline_run.json not found in {run_dir}")

    logger.info(f"Loading run from {run_dir}")

    # --- faces.csv ---
    logger.info(f"  Reading {faces_csv}")
    faces_df = pd.read_csv(faces_csv)
    # Build O(1) lookup: face_id -> cluster_id
    face_id_to_cluster = dict(zip(faces_df["face_id"].astype(int),
                                  faces_df["cluster_id"].astype(int)))

    # --- clusters.csv (prefer base snapshot over merged-overwritten version) ---
    clusters_base_csv = run_dir / "clusters_stage_base.csv"
    clusters_csv = run_dir / "clusters.csv"
    if clusters_base_csv.exists():
        clusters_df = pd.read_csv(clusters_base_csv)
        logger.info(f"  Reading {clusters_base_csv} ({len(clusters_df)} base clusters)")
    elif clusters_csv.exists():
        clusters_df = pd.read_csv(clusters_csv)
        logger.info(f"  Reading {clusters_csv} ({len(clusters_df)} clusters)")
    else:
        clusters_df = pd.DataFrame()

    # --- embeddings (optional) ---
    emb_by_id: dict[int, np.ndarray] = {}
    emb_path   = run_dir / "embeddings.npy"
    emb_id_path = run_dir / "embedding_face_ids.npy"
    if emb_path.exists() and emb_id_path.exists():
        emb_matrix   = np.load(emb_path)
        emb_face_ids = np.load(emb_id_path)
        for fid, row in zip(emb_face_ids, emb_matrix):
            if np.any(row):  # skip zero rows (holdout with no embedding)
                emb_by_id[int(fid)] = row
        logger.info(f"  Loaded {len(emb_by_id)} embeddings from {emb_path}")
    else:
        logger.warning(f"  embeddings.npy not found in {run_dir} — "
                       "analysis views requiring embeddings will be unavailable. "
                       "Re-run the pipeline to generate embeddings.npy.")

    # --- Reconstruct FaceRecord list ---
    faces: list[FaceRecord] = []
    for _, row in faces_df.iterrows():
        fid = int(row["face_id"])
        pose: Optional[tuple] = None
        if "yaw" in row and pd.notna(row.get("yaw")):
            pose = (float(row["yaw"]),
                    float(row.get("pitch", 0.0)),
                    float(row.get("roll", 0.0)))

        emb = emb_by_id.get(fid)
        det_score = float(row["det_score"]) if "det_score" in row and pd.notna(row.get("det_score")) else None
        d10_score = float(row["d10_score"]) if "d10_score" in row and pd.notna(row.get("d10_score")) else None
        rejection_reason = str(row["quality_rejection_reason"]) if "quality_rejection_reason" in row and pd.notna(row.get("quality_rejection_reason")) else None

        face = FaceRecord(
            face_id=fid,
            image_id=str(row.get("image_id", "")),
            bbox=(0, 0, 0, 0),
            embedding_normalized=emb,
            blur_score=float(row.get("blur_score", 0.0)),
            area=float(row.get("area", 0.0)),
            is_core=bool(row.get("is_core", False)),
            image_path=str(row.get("image_path", "")) or None,
            pose=pose,
            det_score=det_score,
            d10_score=d10_score,
            rejection_reason=rejection_reason,
            quality_verdict=_load_verdict(row),
        )
        faces.append(face)

    # --- Reconstruct ClusterResult ---
    # clusters dict: cluster_id -> list of face-list indices (not face_ids)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        cid = face_id_to_cluster.get(face.face_id, -1)
        if cid >= 0:
            clusters[cid].append(i)

    labels = np.array([face_id_to_cluster.get(f.face_id, -1) for f in faces], dtype=int)

    exemplars: dict[int, list[int]] = {}
    if not clusters_df.empty and "exemplar_face_ids" in clusters_df.columns:
        face_id_to_idx = {f.face_id: i for i, f in enumerate(faces)}
        for _, row in clusters_df.iterrows():
            cid = int(row["cluster_id"])
            try:
                ex_ids = ast.literal_eval(str(row["exemplar_face_ids"]))
                exemplars[cid] = [face_id_to_idx[eid] for eid in ex_ids
                                  if eid in face_id_to_idx]
            except Exception:
                exemplars[cid] = clusters[cid][:1]
    else:
        for cid, members in clusters.items():
            exemplars[cid] = members[:1]

    cluster_result = ClusterResult(
        labels=labels,
        clusters=dict(clusters),
        cluster_stats={},
        exemplars=exemplars,
        n_clusters=len(clusters),
        n_noise=int((labels == -1).sum()),
    )

    # --- pipeline_run.json summary ---
    with open(run_json, encoding="utf-8") as fh:
        rec = json.load(fh)
    summary = rec.get("summary", {})
    summary["stages_timing"] = {
        k: v.get("elapsed_s") for k, v in rec.get("stages", {}).items()
    }
    # Promote top-level provenance fields into summary for easy access
    for _key in ("mode", "source_run", "source_type", "config"):
        if _key in rec and _key not in summary:
            summary[_key] = rec[_key]

    # --- merged results (optional) ---
    merged_cluster_result = None
    merge_log = None
    faces_merged_csv = run_dir / "faces_merged.csv"
    clusters_merged_csv = run_dir / "clusters_merged.csv"
    if faces_merged_csv.exists() and clusters_merged_csv.exists():
        merged_cluster_result = _load_cluster_result(faces, faces_merged_csv, clusters_merged_csv)
        logger.info(f"  Loaded merged result: {merged_cluster_result.n_clusters} clusters")
    merge_log_path = run_dir / "merge_log.json"
    if merge_log_path.exists():
        with open(merge_log_path, encoding="utf-8") as fh:
            merge_log = json.load(fh)
        logger.info(f"  Loaded {len(merge_log)} merge decisions")

    merge_metadata = None
    merge_metadata_path = run_dir / "merge_metadata.json"
    if merge_metadata_path.exists():
        with open(merge_metadata_path, encoding="utf-8") as fh:
            merge_metadata = json.load(fh)
        logger.info("  Loaded merge_metadata.json")

    merge_decisions = load_merge_decisions(run_dir)
    if merge_decisions is not None:
        logger.info(f"  Loaded {len(merge_decisions)} merge decisions from merge_decisions.json")

    logger.info(f"  Loaded {len(faces)} faces, {len(clusters)} clusters")
    return PipelineResult(
        faces=faces,
        cluster_result=cluster_result,
        output_dir=run_dir,
        summary=summary,
        merged_cluster_result=merged_cluster_result,
        merge_log=merge_log,
        merge_metadata=merge_metadata,
        merge_decisions=merge_decisions,
    )


_QUALITY_GATES = ("blur", "pose_yaw", "pose_pitch", "area")


def _load_verdict(row: "pd.Series") -> Optional[QualityVerdict]:
    """Reconstruct QualityVerdict from a faces.csv row; return None if no gate columns present."""
    gates = {}
    for gate_name in _QUALITY_GATES:
        val_col = f"quality_{gate_name}_value"
        pass_col = f"quality_{gate_name}_pass"
        if val_col not in row or pass_col not in row:
            continue
        if pd.isna(row.get(val_col)) or pd.isna(row.get(pass_col)):
            continue
        gates[gate_name] = GateResult(
            value=float(row[val_col]),
            threshold=0.0,  # threshold not persisted per-row; available in quality_config.json
            passed=bool(row[pass_col]),
        )
    if not gates:
        return None
    rejection_reason = (
        str(row["quality_rejection_reason"])
        if "quality_rejection_reason" in row and pd.notna(row.get("quality_rejection_reason"))
        else None
    )
    return QualityVerdict(gates=gates, rejection_reason=rejection_reason)


def _load_cluster_result(
    faces: list,
    faces_csv: Path,
    clusters_csv: Path,
) -> "ClusterResult":
    """Reconstruct a ClusterResult from merged CSV files."""
    from collections import defaultdict
    faces_df = pd.read_csv(faces_csv)
    clusters_df = pd.read_csv(clusters_csv)

    face_id_to_cluster = dict(zip(faces_df["face_id"].astype(int),
                                  faces_df["cluster_id"].astype(int)))

    clusters: dict[int, list[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        cid = face_id_to_cluster.get(face.face_id, -1)
        if cid >= 0:
            clusters[cid].append(i)

    labels = np.array([face_id_to_cluster.get(f.face_id, -1) for f in faces], dtype=int)

    exemplars: dict[int, list[int]] = {}
    face_id_to_idx = {f.face_id: i for i, f in enumerate(faces)}
    if not clusters_df.empty and "exemplar_face_ids" in clusters_df.columns:
        for _, row in clusters_df.iterrows():
            cid = int(row["cluster_id"])
            ex_ids = ast.literal_eval(str(row["exemplar_face_ids"]))
            exemplars[cid] = [face_id_to_idx[eid] for eid in ex_ids if eid in face_id_to_idx]
    else:
        for cid, members in clusters.items():
            exemplars[cid] = members[:1]

    return ClusterResult(
        labels=labels,
        clusters=dict(clusters),
        cluster_stats={},
        exemplars=exemplars,
        n_clusters=len(clusters),
        n_noise=int((labels == -1).sum()),
    )
