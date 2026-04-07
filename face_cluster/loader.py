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

from face_cluster.pipeline import PipelineResult
from face_cluster.types import ClusterResult, FaceRecord

logger = logging.getLogger(__name__)


def load_pipeline_result(run_dir: Path) -> PipelineResult:
    """Reconstruct a PipelineResult from a completed run directory.

    Never re-runs any model.  Embeddings are loaded from embeddings.npy if
    present; otherwise FaceRecord.embedding_normalized is left as None and
    analysis views that require embeddings (UMAP, distances) will degrade
    gracefully.

    Args:
        run_dir: Directory produced by FaceClusteringPipeline.run().

    Returns:
        PipelineResult with faces, cluster_result, output_dir, summary.

    Raises:
        FileNotFoundError: if faces.csv or pipeline_run.json are missing.
    """
    run_dir = Path(run_dir)

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

    # --- clusters.csv ---
    clusters_csv = run_dir / "clusters.csv"
    clusters_df = pd.read_csv(clusters_csv) if clusters_csv.exists() else pd.DataFrame()
    logger.info(f"  Reading {clusters_csv} ({len(clusters_df)} clusters)")

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
        faces.append(FaceRecord(
            face_id=fid,
            image_id=str(row.get("image_id", "")),
            bbox=(0, 0, 0, 0),
            embedding_normalized=emb,
            blur_score=float(row.get("blur_score", 0.0)),
            area=float(row.get("area", 0.0)),
            is_core=bool(row.get("is_core", False)),
            image_path=str(row.get("image_path", "")) or None,
            pose=pose,
        ))

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

    logger.info(f"  Loaded {len(faces)} faces, {len(clusters)} clusters")
    return PipelineResult(
        faces=faces,
        cluster_result=cluster_result,
        output_dir=run_dir,
        summary=summary,
    )
