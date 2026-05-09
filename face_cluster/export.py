"""Export clustering results to CSV and JSON."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from face_cluster.types import FaceRecord, ClusterResult
from face_cluster.config import PipelineConfig

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


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
        row = {
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
            # Observability fields (spec 012)
            "det_score": face.det_score,
            "d10_score": face.d10_score,
            "quality_rejection_reason": face.rejection_reason,
        }
        verdict = face.quality_verdict
        if verdict is not None:
            for gate_name, gate in verdict.gates.items():
                row[f"quality_{gate_name}_value"] = gate.value
                row[f"quality_{gate_name}_pass"] = gate.passed
        rows.append(row)

    faces_df = pd.DataFrame(rows)
    faces_csv = output_dir / "faces.csv"
    faces_df.to_csv(faces_csv, index=False)
    logger.info(f"Wrote {len(faces_df)} rows to {faces_csv}")

    # clusters.csv
    cluster_rows = []
    for cluster_id, face_indices in cluster_result.clusters.items():
        # Remap exemplar graph-local indices -> face_ids (same mapping as clusters above)
        raw_exemplar_indices = cluster_result.exemplars.get(cluster_id, [])
        exemplar_face_ids = []
        for graph_node in raw_exemplar_indices:
            face_list_idx = core_indices[graph_node] if core_indices is not None else graph_node
            exemplar_face_ids.append(faces[face_list_idx].face_id)

        stats = cluster_result.cluster_stats.get(cluster_id, {})
        cluster_rows.append({
            "cluster_id": cluster_id,
            "size": len(face_indices),
            "exemplar_face_ids": json.dumps(exemplar_face_ids),
            "diameter": stats.get("diameter", None),
            "origin": "base",
            "parent_cluster_ids": "[]",
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


def export_merged_results(
    faces: List[FaceRecord],
    merged_cluster_result: ClusterResult,
    merge_log: List[Dict],
    output_dir: Path,
    core_indices: Optional[List[int]] = None,
    merge_metadata: Optional[Dict] = None,
) -> None:
    """Export merged clustering results alongside the base export.

    Writes:
        faces_merged.csv         - face_id + merged cluster_id per face
        clusters_merged.csv      - merged cluster summary (legacy alias for clusters.csv)
        clusters.csv             - final cluster state (overwritten with merged provenance)
        clusters_stage_base.csv  - renamed pre-merge clusters.csv (provenance snapshot)
        merge_log.json           - full merge decision log from ConservativeMerger
        merge_metadata.json      - cluster thresholds and run statistics
    """
    output_dir = Path(output_dir)

    # Snapshot pre-merge clusters.csv before overwriting
    pre_merge_csv = output_dir / "clusters.csv"
    if pre_merge_csv.exists():
        pre_merge_csv.rename(output_dir / "clusters_stage_base.csv")
        logger.info("Renamed clusters.csv -> clusters_stage_base.csv (pre-merge snapshot)")

    face_list_idx_to_cluster: Dict[int, int] = {}
    for cluster_id, graph_nodes in merged_cluster_result.clusters.items():
        for node in graph_nodes:
            fl_idx = core_indices[node] if core_indices is not None else node
            face_list_idx_to_cluster[fl_idx] = cluster_id

    faces_rows = [
        {"face_id": face.face_id, "cluster_id": face_list_idx_to_cluster.get(i, -1)}
        for i, face in enumerate(faces)
    ]
    pd.DataFrame(faces_rows).to_csv(output_dir / "faces_merged.csv", index=False)
    logger.info(f"Wrote {len(faces_rows)} rows to faces_merged.csv")

    # Build parent-cluster map from merge_log union-find
    parent_map = _build_parent_map(merge_log)

    cluster_rows = []
    for cluster_id, graph_nodes in merged_cluster_result.clusters.items():
        raw_exemplars = merged_cluster_result.exemplars.get(cluster_id, [])
        exemplar_face_ids = []
        for node in raw_exemplars:
            fl_idx = core_indices[node] if core_indices is not None else node
            exemplar_face_ids.append(faces[fl_idx].face_id)
        stats = merged_cluster_result.cluster_stats.get(cluster_id, {})
        parents = parent_map.get(cluster_id, [])
        origin = "auto_merge" if parents else "base"
        cluster_rows.append({
            "cluster_id": cluster_id,
            "size": len(graph_nodes),
            "exemplar_face_ids": json.dumps(exemplar_face_ids),
            "diameter": stats.get("diameter", None),
            "origin": origin,
            "parent_cluster_ids": json.dumps(parents),
        })

    merged_df = pd.DataFrame(cluster_rows)
    merged_df.to_csv(output_dir / "clusters_merged.csv", index=False)
    merged_df.to_csv(output_dir / "clusters.csv", index=False)
    logger.info(f"Wrote {len(cluster_rows)} merged clusters to clusters.csv / clusters_merged.csv")

    with open(output_dir / "merge_log.json", "w", encoding="utf-8") as f:
        json.dump(merge_log, f, cls=_NumpyEncoder, indent=2)
    logger.info(f"Wrote {len(merge_log)} merge decisions to merge_log.json")

    if merge_metadata is not None:
        with open(output_dir / "merge_metadata.json", "w", encoding="utf-8") as f:
            json.dump(merge_metadata, f, cls=_NumpyEncoder, indent=2)
        logger.info("Wrote merge_metadata.json")


def _build_parent_map(merge_log: List[Dict]) -> Dict[int, List[int]]:
    """Union-find over actually_merged=True entries to build {final_id: [original_ids]}.

    When cluster A and B merge into A, the parent_map records {A: [original A members, B]}.
    """
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
        if not entry.get("actually_merged"):
            continue
        a, b = entry.get("cluster_a"), entry.get("cluster_b")
        if a is None or b is None:
            continue
        if a not in originals:
            originals[a] = [a]
        if b not in originals:
            originals[b] = [b]
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
            originals[ra] = originals.get(ra, [ra]) + originals.get(rb, [rb])

    # Build final map: root -> list of merged-in original IDs (excluding self)
    result: Dict[int, List[int]] = {}
    for cid in originals:
        root = _find(cid)
        members = originals.get(root, [])
        if len(members) > 1:
            result[root] = sorted(set(members) - {root})

    return result


# ---------------------------------------------------------------------------
# merge_decisions.json  — user approve/reject decisions (writer-owns contract)
# ---------------------------------------------------------------------------
# Schema: List of entries, each:
#   cluster_a       int   — first cluster ID (from base clustering)
#   cluster_b       int   — second cluster ID (from base clustering)
#   decision        str   — "approve" | "reject"
#   n_gates_passed  int   — number of heuristic gates that passed
#   exemplar_dist   float — min exemplar distance between the two clusters
#   threshold_used  float — exemplar gate threshold that was in effect
#   support         int   — number of cross-cluster pairs below threshold
#   required_support int  — minimum support required by heuristic
#   margin_gap      float | null
#   post_diameter   float
#   run_id          str   — output_folder name of the run (traceability)
#   timestamp       str   — ISO-8601 when the decision was saved

_MERGE_DECISIONS_FILE = "merge_decisions.json"


def save_merge_decisions(decisions: List[Dict], output_dir: Path) -> None:
    """Persist user merge approve/reject decisions to merge_decisions.json.

    Args:
        decisions: List of decision dicts (see schema above).
        output_dir: Run output directory.
    """
    output_dir = Path(output_dir)
    path = output_dir / _MERGE_DECISIONS_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, cls=_NumpyEncoder, indent=2)
    logger.info(f"Saved {len(decisions)} merge decisions to {path}")


def load_merge_decisions(run_dir: Path) -> Optional[List[Dict]]:
    """Load merge decisions from a run directory.

    Args:
        run_dir: Run output directory.

    Returns:
        List of decision dicts, or None if the file does not exist.

    Raises:
        ValueError: If the file exists but cannot be parsed as JSON.
    """
    path = Path(run_dir) / _MERGE_DECISIONS_FILE
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {_MERGE_DECISIONS_FILE} in {run_dir}: {exc}") from exc


# ---------------------------------------------------------------------------
# merge_features.parquet  — ML feature vectors + human labels (writer-owns contract)
# ---------------------------------------------------------------------------
# Schema: one row per labeled cluster pair.
#   cluster_a         int   — first cluster ID (smaller)
#   cluster_b         int   — second cluster ID (larger)
#   label             int   — 1=approve (merge), 0=reject, NaN=unlabeled
#   run_id            str   — output folder name (traceability)
#   timestamp         str   — ISO-8601 when saved
#   feature_version   int   — FeatureComputer.VERSION
#   <feature columns> float — all ClusterPairFeatures fields

_MERGE_FEATURES_FILE = "merge_features.parquet"


def save_merge_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist merge feature vectors (with labels) to merge_features.parquet.

    Args:
        df: DataFrame with cluster_a, cluster_b, label, run_id, timestamp,
            feature_version, and all feature columns.
        output_dir: Run output directory.
    """
    path = Path(output_dir) / _MERGE_FEATURES_FILE
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} merge feature rows to {path}")


def load_merge_features(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load merge feature vectors from a run directory.

    Returns:
        DataFrame or None if the file does not exist.
    """
    path = Path(run_dir) / _MERGE_FEATURES_FILE
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# candidate_pairs.parquet  — top-N cluster pair features, auto-saved each run
# ---------------------------------------------------------------------------
# Schema: one row per candidate cluster pair (no labels).
#   cluster_a         int   — first cluster ID (smaller)
#   cluster_b         int   — second cluster ID (larger)
#   feature_version   int   — FeatureComputer.VERSION
#   saved_at          str   — ISO-8601 when saved
#   <feature columns> float — all ClusterPairFeatures fields
#
# Saved automatically after the exemplars stage (top 300, max exemplar dist 0.80).
# Use load_candidate_pairs() to load for what-if threshold analysis.

_CANDIDATE_PAIRS_FILE = "candidate_pairs.parquet"


def save_candidate_pairs(df: pd.DataFrame, output_dir: Path) -> None:
    """Save top-N candidate pair features to candidate_pairs.parquet.

    Args:
        df: DataFrame with cluster_a, cluster_b, feature_version, saved_at,
            and all ClusterPairFeatures columns.
        output_dir: Run output directory.
    """
    path = Path(output_dir) / _CANDIDATE_PAIRS_FILE
    df.to_parquet(path, index=False)
    logger.info("Saved %d candidate pair rows to %s", len(df), path)


def load_candidate_pairs(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load candidate pair features from a run directory.

    Returns:
        DataFrame or None if candidate_pairs.parquet does not exist.
    """
    path = Path(run_dir) / _CANDIDATE_PAIRS_FILE
    if not path.exists():
        return None
    return pd.read_parquet(path)
