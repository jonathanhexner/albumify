"""Export clustering results to CSV and JSON."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from face_cluster.types import FaceRecord, ClusterResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


def export_results(
    faces: List[FaceRecord],
    cluster_result: ClusterResult,
    crop_manifest: Dict[int, Path],
    output_dir: Path,
    config: PipelineConfig,
    source_album: str,
    core_indices: Optional[List[int]] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Export clustering results to CSV and JSON.

    Writes:
        faces.csv            - one row per face
        clusters.csv         - one row per cluster
        export_summary.json  - metadata
        embeddings.npy       - float32 array (n_faces x 512), L2-normalised embeddings
        embedding_face_ids.npy - int32 array (n_faces,), face_id per row in embeddings.npy

    Args:
        faces: All FaceRecord objects (core + holdout)
        cluster_result: Result from clustering stage
        crop_manifest: Dict mapping face_id -> crop Path (from save_crops())
        output_dir: Directory to write files into
        config: Pipeline config (saved to summary)
        source_album: Source image directory path (for traceability)
        run_id: Optional run identifier (defaults to ISO timestamp)

    Returns:
        Path to output_dir
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build face list-index -> cluster_id mapping.
    # cluster_result.clusters stores graph-node indices (0..n_core-1).
    # core_indices[graph_node] gives the index into the full faces list.
    face_list_idx_to_cluster: Dict[int, int] = {}
    for cluster_id, graph_node_indices in cluster_result.clusters.items():
        for graph_node in graph_node_indices:
            if core_indices is not None:
                face_list_idx = core_indices[graph_node]
            else:
                face_list_idx = graph_node  # fallback: assume all faces are core
            face_list_idx_to_cluster[face_list_idx] = cluster_id

    # faces.csv
    rows = []
    for i, face in enumerate(faces):
        if face.image_path is None:
            logger.warning(f"face_id {face.face_id} has null image_path")
        yaw = pitch = roll = None
        if face.pose is not None:
            yaw, pitch, roll = face.pose
        rows.append({
            "face_id": face.face_id,
            "image_path": face.image_path,
            "image_id": face.image_id,
            "crop_path": str(crop_manifest.get(face.face_id, "")),
            "cluster_id": face_list_idx_to_cluster.get(i, -1),
            "is_core": face.is_core,
            "blur_score": face.blur_score,
            "area": face.area,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        })

    faces_df = pd.DataFrame(rows)
    faces_csv = output_dir / "faces.csv"
    faces_df.to_csv(faces_csv, index=False)
    logger.info(f"Wrote {len(faces_df)} rows to {faces_csv}")

    # clusters.csv
    cluster_rows = []
    for cluster_id, face_indices in cluster_result.clusters.items():
        exemplar_ids = cluster_result.exemplars.get(cluster_id, [])
        stats = cluster_result.cluster_stats.get(cluster_id, {})
        cluster_rows.append({
            "cluster_id": cluster_id,
            "size": len(face_indices),
            "exemplar_face_ids": json.dumps(exemplar_ids),
            "diameter": stats.get("diameter", None),
        })

    clusters_df = pd.DataFrame(cluster_rows)
    clusters_csv = output_dir / "clusters.csv"
    clusters_df.to_csv(clusters_csv, index=False)
    logger.info(f"Wrote {len(clusters_df)} clusters to {clusters_csv}")

    # embeddings.npy + embedding_face_ids.npy
    # Rows aligned with faces list order.  Holdout faces with no embedding get zeros.
    EMB_DIM = 512
    emb_matrix = np.zeros((len(faces), EMB_DIM), dtype=np.float32)
    emb_face_ids = np.array([f.face_id for f in faces], dtype=np.int32)
    for i, face in enumerate(faces):
        if face.embedding_normalized is not None:
            emb_matrix[i] = face.embedding_normalized.astype(np.float32)
    np.save(output_dir / "embeddings.npy", emb_matrix)
    np.save(output_dir / "embedding_face_ids.npy", emb_face_ids)
    logger.info(f"Wrote embeddings ({emb_matrix.shape}) to {output_dir / 'embeddings.npy'}")

    # export_summary.json
    summary = {
        "source_album": str(source_album),
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "n_faces": len(faces),
        "n_core": sum(1 for f in faces if f.is_core),
        "n_clusters": cluster_result.n_clusters,
        "n_noise": cluster_result.n_noise,
        "config": {
            k: v for k, v in vars(config).items()
            if not k.startswith("_")
        },
    }
    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote export summary to {summary_path}")

    return output_dir
