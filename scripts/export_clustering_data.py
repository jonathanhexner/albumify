"""
Export face clustering data for ML training.

Runs mutual kNN clustering pipeline and exports:
1. faces.csv - one row per face with metadata
2. clusters.csv - one row per cluster with statistics
3. candidate_pairs.csv - cluster pair features for merge candidates

Usage:
    python scripts/export_clustering_data.py \
        --embeddings results/face_clustering_benchmark/embeddings_*.npy \
        --output results/face_clustering_training/dataset1 \
        --k 5 \
        --distance-threshold 0.35 \
        --candidate-threshold 0.45
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import sys

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import (
    PipelineConfig,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
    ConservativeMerger,
    FaceRecord,
    GraphResult,
    ClusterResult,
    FeatureComputer,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to both console and file."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'export_{timestamp}.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")
    return log_file


def get_face_crop(face_idx: int, crops_dir: Path) -> Optional[np.ndarray]:
    """Load face crop image from benchmark results."""
    for pattern in [f"face_{face_idx:04d}_aligned.jpg", f"face_{face_idx:04d}.jpg"]:
        crop_path = crops_dir / pattern
        if crop_path.exists():
            return np.array(Image.open(crop_path))
    return None


def load_embeddings_and_metadata(
    embeddings_path: Path,
    estimate_pose: bool = True,
    device: str = 'cpu'
) -> Tuple[List[FaceRecord], Path]:
    """
    Load pre-computed embeddings and metadata from benchmark results.

    Args:
        embeddings_path: Path to embeddings_*.npy file
        estimate_pose: Whether to compute pose from face crops (requires sixdrepnet)
        device: Device for pose estimation ('cpu' or 'cuda')

    Returns:
        faces: List of FaceRecord objects
        crops_dir: Directory containing face crop images
    """
    logger.info(f"Loading embeddings from: {embeddings_path}")

    # Load embeddings
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded {embeddings.shape[0]} embeddings")

    # Load metadata JSON
    json_file = embeddings_path.parent / embeddings_path.name.replace("embeddings_", "benchmark_").replace(".npy", ".json")
    if json_file.exists():
        with open(json_file) as f:
            benchmark_data = json.load(f)
        face_metadata = benchmark_data.get('face_metadata', [])
        logger.info(f"Loaded metadata for {len(face_metadata)} faces")

        if len(face_metadata) != len(embeddings):
            logger.warning(f"Metadata count ({len(face_metadata)}) != embeddings count ({len(embeddings)})")
            face_metadata = []
    else:
        logger.warning(f"Metadata file not found: {json_file.name}")
        face_metadata = []

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create FaceRecord objects
    faces = []
    crops_dir = embeddings_path.parent / "face_crops"

    for i in range(len(embeddings)):
        aligned_face = get_face_crop(i, crops_dir)

        # Use metadata if available
        if i < len(face_metadata):
            meta = face_metadata[i]
            # Handle None image_path
            if meta.get('image_path') is not None:
                image_id = Path(meta['image_path']).name
                image_path = meta['image_path']
            else:
                image_id = f"face_{i:04d}"
                image_path = None

            face_index = meta.get('face_index')
            bbox_data = meta.get('bbox', {})
            bbox = (
                bbox_data.get('x_px', 0.0),
                bbox_data.get('y_px', 0.0),
                bbox_data.get('w_px', 112.0),
                bbox_data.get('h_px', 112.0)
            )
        else:
            image_id = f"face_{i:04d}"
            image_path = None
            face_index = None
            bbox = (0.0, 0.0, 112.0, 112.0)

        face = FaceRecord(
            face_id=i,
            image_id=image_id,
            bbox=bbox,
            aligned_face=aligned_face,
            embedding=embeddings[i],
            embedding_normalized=embeddings[i],
            pose=(0.0, 0.0, 0.0),  # Will be computed if estimate_pose=True
            blur_score=100.0,      # Will be computed in quality gating
            area=bbox[2] * bbox[3] if isinstance(bbox, tuple) else 12544.0,
            is_core=False,
            image_path=image_path,
            face_index=face_index
        )
        faces.append(face)

    logger.info(f"Created {len(faces)} FaceRecord objects")
    logger.info(f"Face crops loaded: {sum(1 for f in faces if f.aligned_face is not None)} / {len(faces)}")

    # Show unique source images
    if face_metadata:
        unique_paths = [m.get('image_path') for m in face_metadata if m.get('image_path') is not None]
        if unique_paths:
            unique_images = len(set(Path(p).name for p in unique_paths))
            logger.info(f"From {unique_images} unique source images")

    return faces, crops_dir


def run_clustering_pipeline(
    faces: List[FaceRecord],
    config: PipelineConfig,
    estimate_pose: bool = True
) -> Tuple[List[int], GraphResult, ClusterResult, np.ndarray, ClusterResult, List[Dict]]:
    """
    Run clustering pipeline: quality gating → kNN graph → connected components → exemplars → merge.

    Args:
        faces: List of FaceRecord objects
        config: Pipeline configuration
        estimate_pose: Whether to compute pose from face crops

    Returns:
        core_indices: Indices of faces that passed quality gating
        graph_result: kNN graph result
        cluster_result: Clustering result with exemplars (post-merge)
        distance_matrix: Pairwise distance matrix for core faces
        pre_merge_result: Clustering result before merge (None if merge disabled)
        merge_log: List of merge decision dicts
    """
    logger.info("Running clustering pipeline...")

    # Stage 1: Quality gating
    logger.info("Stage 1: Quality gating")
    gater = QualityGater(
        config,
        use_pose_estimation=estimate_pose,
        device='cpu'
    )

    # Compute blur scores
    faces = gater.compute_blur_scores(faces)

    # Optionally compute pose
    if estimate_pose:
        logger.info("Computing pose from face crops (this may take a while)...")
        faces = gater.compute_pose_scores(faces)
        poses_computed = sum(1 for f in faces if f.pose is not None and f.pose != (0.0, 0.0, 0.0))
        logger.info(f"Pose computed for {poses_computed}/{len(faces)} faces")

    # Filter by blur + pose
    core_indices = []
    holdout_indices = []

    for i, face in enumerate(faces):
        passes_blur = face.blur_score >= config.blur_min
        passes_pose = True

        if estimate_pose and face.pose is not None:
            yaw, pitch, roll = face.pose
            passes_pose = (
                abs(yaw) <= config.yaw_max and
                abs(pitch) <= config.pitch_max and
                abs(roll) <= config.roll_max
            )

        if passes_blur and passes_pose:
            core_indices.append(i)
            face.is_core = True
        else:
            holdout_indices.append(i)
            face.is_core = False

    logger.info(f"Quality gating results: core={len(core_indices)}, holdout={len(holdout_indices)}")

    if len(core_indices) == 0:
        raise ValueError("No faces passed quality gating!")

    # Stage 2: Build distance matrix
    logger.info("Stage 2: Building distance matrix")
    graph_builder = KNNGraphBuilder(config)
    distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)
    logger.info(f"Distance matrix shape: {distance_matrix.shape}")

    # Stage 3: Build mutual kNN graph
    logger.info("Stage 3: Building mutual kNN graph")
    graph_result = graph_builder.build_mutual_knn_graph(
        distance_matrix,
        config.K,
        config.distance_threshold
    )
    logger.info(f"Graph: {graph_result.G.number_of_nodes()} nodes, {graph_result.G.number_of_edges()} edges")

    # Stage 4: Connected components clustering
    logger.info("Stage 4: Connected components clustering")
    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)
    logger.info(f"Clustering: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # Stage 5: Exemplar selection
    logger.info("Stage 5: D10 exemplar selection")
    exemplar_selector = D10ExemplarSelector(config)
    cluster_result = exemplar_selector.select_exemplars(cluster_result, graph_result)

    total_exemplars = sum(len(exs) for exs in cluster_result.exemplars.values())
    logger.info(f"Selected {total_exemplars} exemplars across {len(cluster_result.exemplars)} clusters")

    # Stage 6: Conservative merge (if enabled)
    merge_log = []
    pre_merge_result = None

    if config.merge_enabled:
        logger.info("Stage 6: Conservative merge")
        initial_clusters = cluster_result.n_clusters

        # Save pre-merge state
        pre_merge_result = cluster_result

        # Merge with logging
        merger = ConservativeMerger(config)
        cluster_result, merge_log = merger.merge_clusters_with_logging(cluster_result, graph_result)
        logger.info(f"Merging: {initial_clusters} → {cluster_result.n_clusters} clusters (merged {initial_clusters - cluster_result.n_clusters})")
        logger.info(f"Logged {len(merge_log)} merge decisions")
    else:
        logger.info("Stage 6: Skipping merge (disabled in config)")

    return core_indices, graph_result, cluster_result, distance_matrix, pre_merge_result, merge_log


def export_knn_graph(
    graph_result: GraphResult,
    faces: List[FaceRecord],
    core_indices: List[int],
    config: PipelineConfig,
    output_dir: Path
):
    """
    Export kNN graph structure for diagnostics.

    Saves:
    - Each face's K nearest neighbors (IDs and distances)
    - Edge list (mutual kNN connections)
    - Graph statistics

    Args:
        graph_result: kNN graph result
        faces: All faces
        core_indices: Core face indices
        config: Pipeline config
        output_dir: Output directory
    """
    logger.info("Exporting kNN graph structure...")

    distance_matrix = graph_result.distance_matrix
    G = graph_result.G

    # 1. For each face, find K nearest neighbors (in full dataset, not just graph)
    knn_data = []
    for i, node_idx in enumerate(core_indices):
        face_id = faces[node_idx].face_id

        # Get distances to all other core faces
        distances = distance_matrix[i, :]

        # Find K nearest (excluding self)
        k_nearest_indices = np.argsort(distances)[1:config.K+1]  # Skip first (self)
        k_nearest_face_ids = [faces[core_indices[idx]].face_id for idx in k_nearest_indices]
        k_nearest_distances = [float(distances[idx]) for idx in k_nearest_indices]

        knn_data.append({
            'face_id': face_id,
            'k_nearest_ids': k_nearest_face_ids,
            'k_nearest_distances': k_nearest_distances
        })

    # 2. Extract edge list from graph (mutual kNN edges that passed threshold)
    edges = []
    for edge in G.edges():
        node_a, node_b = edge
        face_a_id = faces[core_indices[node_a]].face_id
        face_b_id = faces[core_indices[node_b]].face_id
        distance = distance_matrix[node_a, node_b]

        edges.append({
            'face_a': face_a_id,
            'face_b': face_b_id,
            'distance': float(distance)
        })

    # 3. Save as JSON
    knn_graph_data = {
        'config': {
            'K': config.K,
            'distance_threshold': config.distance_threshold,
        },
        'statistics': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_core_faces': len(core_indices)
        },
        'knn_neighbors': knn_data,
        'edges': edges
    }

    knn_path = output_dir / 'knn_graph.json'
    with open(knn_path, 'w') as f:
        json.dump(knn_graph_data, f, indent=2)

    logger.info(f"Exported kNN graph to: {knn_path}")
    logger.info(f"  - {len(knn_data)} faces with K={config.K} neighbors each")
    logger.info(f"  - {len(edges)} mutual kNN edges")

    return knn_path


def compute_cluster_stats(
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    faces: List[FaceRecord],
    core_indices: List[int],
    feature_computer: FeatureComputer
) -> Dict[int, Dict[str, Any]]:
    """
    Compute statistics for each cluster using FeatureComputer.

    Args:
        cluster_result: Clustering result
        distance_matrix: Pairwise distance matrix
        faces: All faces
        core_indices: Core face indices
        feature_computer: FeatureComputer instance

    Returns:
        Dict mapping cluster_id to stats dict with keys:
            - diameter: max pairwise distance within cluster
            - T_A: P90 of exemplar pairwise distances
            - mean_blur: mean blur score
            - face_ids: list of face IDs in cluster
            - frontal_frac: fraction of frontal faces (V2)
            - mean_yaw: mean yaw angle (V2)
            - mean_pitch: mean pitch angle (V2)
    """
    logger.info("Computing cluster statistics...")

    cluster_stats = {}

    for cluster_id, cluster_nodes in cluster_result.clusters.items():
        # Get exemplars
        exemplar_nodes = cluster_result.exemplars.get(cluster_id, [])

        # Compute stats using FeatureComputer
        stats = feature_computer.compute_cluster_stats(
            cluster_nodes=cluster_nodes,
            exemplar_nodes=exemplar_nodes,
            distance_matrix=distance_matrix,
            face_records=faces,
            core_indices=core_indices
        )

        cluster_stats[cluster_id] = stats

    # Compute T_global (median of all T_A values)
    T_A_values = [stats['T_A'] for stats in cluster_stats.values() if stats['T_A'] > 0]
    T_global = float(np.median(T_A_values)) if len(T_A_values) > 0 else 0.0

    logger.info(f"Cluster stats: T_global={T_global:.3f}")

    return cluster_stats, T_global


def generate_candidate_pairs(
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    candidate_threshold: float = 0.45
) -> List[Tuple[int, int, float]]:
    """
    Generate candidate cluster pairs filtered by exemplar distance.

    Args:
        cluster_result: Clustering result with exemplars
        distance_matrix: Pairwise distance matrix
        candidate_threshold: Max min_exemplar_dist for candidates

    Returns:
        List of (cluster_id_1, cluster_id_2, min_exemplar_dist) sorted by distance
    """
    logger.info(f"Generating candidate pairs (threshold={candidate_threshold})...")

    candidates = []
    cluster_ids = sorted(cluster_result.clusters.keys())

    for i, cluster_id_a in enumerate(cluster_ids):
        for cluster_id_b in cluster_ids[i+1:]:
            # Get exemplars
            exemplars_a = cluster_result.exemplars.get(
                cluster_id_a,
                cluster_result.clusters[cluster_id_a]  # Use all if no exemplars
            )
            exemplars_b = cluster_result.exemplars.get(
                cluster_id_b,
                cluster_result.clusters[cluster_id_b]
            )

            # Compute min exemplar distance
            exemplar_dists = distance_matrix[np.ix_(exemplars_a, exemplars_b)]
            min_exemplar_dist = float(exemplar_dists.min())

            # Filter by threshold
            if min_exemplar_dist < candidate_threshold:
                candidates.append((cluster_id_a, cluster_id_b, min_exemplar_dist))

    # Sort by distance (ascending)
    candidates.sort(key=lambda x: x[2])

    logger.info(f"Found {len(candidates)} candidate pairs (from {len(cluster_ids)} clusters)")

    return candidates


def compute_pair_features(
    cluster_id_a: int,
    cluster_id_b: int,
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    cluster_stats: Dict[int, Dict[str, Any]],
    T_global: float,
    feature_computer: FeatureComputer
) -> Dict[str, float]:
    """
    Compute all features for a cluster pair using FeatureComputer.

    V1 Features (12):
    - min_exemplar_dist, p10_cross_dist, p50_cross_dist, support_fraction
    - diameter_ratio, cluster_size_min, cluster_size_ratio
    - T_A, T_B, T_local, T_global

    V2 Features (17 = V1 + 5 new):
    - frontal_frac_A, frontal_frac_B
    - pose_diff (Euclidean distance of mean yaw/pitch)
    - min_exemplar_dist_x_pose, p50_cross_dist_x_pose (interaction terms)

    Args:
        cluster_id_a: First cluster ID
        cluster_id_b: Second cluster ID
        cluster_result: Clustering result
        distance_matrix: Pairwise distance matrix
        cluster_stats: Pre-computed cluster statistics
        T_global: Global threshold
        feature_computer: FeatureComputer instance

    Returns:
        Dict of feature name -> value
    """
    # Compute features using FeatureComputer
    features_obj = feature_computer.compute_pair_features(
        cluster_id_a=cluster_id_a,
        cluster_id_b=cluster_id_b,
        cluster_result=cluster_result,
        distance_matrix=distance_matrix,
        cluster_stats=cluster_stats,
        T_global=T_global
    )

    # Convert to dict (automatically excludes None values)
    return features_obj.to_dict()


def export_csvs(
    faces: List[FaceRecord],
    core_indices: List[int],
    cluster_result: ClusterResult,
    cluster_stats: Dict[int, Dict[str, Any]],
    pair_features: List[Dict[str, float]],
    output_dir: Path,
    embeddings_path: Path,
    pre_merge_result: ClusterResult = None,
    merge_log: List[Dict] = None
):
    """
    Export CSV files: faces, clusters, candidate_pairs, and optionally pre-merge data.

    Args:
        faces: All faces
        core_indices: Core face indices
        cluster_result: Clustering result (post-merge)
        cluster_stats: Cluster statistics
        pair_features: List of feature dicts for each pair
        output_dir: Output directory
        embeddings_path: Path to source embeddings file (for face_crops lookup)
        pre_merge_result: Clustering result before merge (optional)
        merge_log: List of merge decision dicts (optional)
    """
    logger.info(f"Exporting CSVs to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. faces.csv
    # Build reverse mapping: face_id -> cluster_id (post-merge)
    face_to_cluster = {}
    for cid, cluster_nodes in cluster_result.clusters.items():
        for node_idx in cluster_nodes:
            face_idx = core_indices[node_idx]
            face_to_cluster[face_idx] = cid

    # Build pre-merge mapping if available
    pre_face_to_cluster = {}
    if pre_merge_result is not None:
        for cid, cluster_nodes in pre_merge_result.clusters.items():
            for node_idx in cluster_nodes:
                face_idx = core_indices[node_idx]
                pre_face_to_cluster[face_idx] = cid

    faces_data = []
    for i, face in enumerate(faces):
        yaw, pitch, roll = face.pose if face.pose else (None, None, None)

        # Look up cluster IDs (-1 for noise/holdout)
        cluster_id = face_to_cluster.get(i, -1)
        pre_merge_cluster_id = pre_face_to_cluster.get(i, -1) if pre_merge_result is not None else -1

        faces_data.append({
            'face_id': face.face_id,
            'image_path': face.image_path or '',
            'cluster_id': cluster_id,
            'pre_merge_cluster_id': pre_merge_cluster_id,
            'bbox_x': face.bbox[0],
            'bbox_y': face.bbox[1],
            'bbox_w': face.bbox[2],
            'bbox_h': face.bbox[3],
            'blur_score': face.blur_score,
            'pose_yaw': yaw,
            'pose_pitch': pitch,
            'pose_roll': roll,
            'is_core': face.is_core,
            'embedding_index': face.face_id,
        })

    faces_df = pd.DataFrame(faces_data)
    faces_path = output_dir / 'faces.csv'
    faces_df.to_csv(faces_path, index=False)
    logger.info(f"Exported {len(faces_df)} faces to: {faces_path}")

    # 2. clusters.csv
    clusters_data = []
    for cluster_id, cluster_nodes in cluster_result.clusters.items():
        exemplar_nodes = cluster_result.exemplars.get(cluster_id, [])
        exemplar_face_ids = [faces[core_indices[n]].face_id for n in exemplar_nodes]
        face_ids = cluster_stats[cluster_id]['face_ids']

        row = {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_nodes),
            'exemplar_ids': ','.join(map(str, exemplar_face_ids)),
            'diameter': cluster_stats[cluster_id]['diameter'],
            'T_A': cluster_stats[cluster_id]['T_A'],
            'mean_blur': cluster_stats[cluster_id]['mean_blur'],
            'face_ids': ','.join(map(str, face_ids)),
        }

        # Add V2 features if available
        if cluster_stats[cluster_id].get('frontal_frac') is not None:
            row['frontal_frac'] = cluster_stats[cluster_id]['frontal_frac']
            row['mean_yaw'] = cluster_stats[cluster_id]['mean_yaw']
            row['mean_pitch'] = cluster_stats[cluster_id]['mean_pitch']

        clusters_data.append(row)

    clusters_df = pd.DataFrame(clusters_data)
    clusters_path = output_dir / 'clusters.csv'
    clusters_df.to_csv(clusters_path, index=False)
    logger.info(f"Exported {len(clusters_df)} clusters to: {clusters_path}")

    # 3. candidate_pairs.csv
    pairs_df = pd.DataFrame(pair_features)
    pairs_path = output_dir / 'candidate_pairs.csv'
    pairs_df.to_csv(pairs_path, index=False)
    logger.info(f"Exported {len(pairs_df)} candidate pairs to: {pairs_path}")

    # Export summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'embeddings_source': str(embeddings_path),  # Store source path for face_crops lookup
        'embeddings_dir': str(embeddings_path.parent),  # Directory containing embeddings and face_crops
        'n_faces': len(faces),
        'n_core': len(core_indices),
        'n_holdout': len(faces) - len(core_indices),
        'n_clusters': cluster_result.n_clusters,
        'n_noise': cluster_result.n_noise,
        'n_candidate_pairs': len(pair_features),
        'files': {
            'faces': faces_path.name,
            'clusters': clusters_path.name,
            'pairs': pairs_path.name,
        }
    }

    # 4. pre_merge_clusters.csv (if available)
    if pre_merge_result is not None:
        logger.info("Exporting pre-merge clustering state...")

        # Save pre-merge cluster statistics
        pre_merge_clusters_data = []
        for cluster_id, cluster_nodes in pre_merge_result.clusters.items():
            # Get diameter from pre-computed stats
            diameter = pre_merge_result.cluster_stats.get(cluster_id, {}).get('diameter', 0.0)

            exemplar_nodes = pre_merge_result.exemplars.get(cluster_id, [])
            exemplar_face_ids = [faces[core_indices[n]].face_id for n in exemplar_nodes]
            face_ids = [faces[core_indices[n]].face_id for n in cluster_nodes]

            pre_merge_clusters_data.append({
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_nodes),
                'diameter': diameter,
                'exemplar_ids': ','.join(map(str, exemplar_face_ids)),
                'face_ids': ','.join(map(str, face_ids)),
            })

        pre_merge_clusters_df = pd.DataFrame(pre_merge_clusters_data)
        pre_merge_path = output_dir / 'pre_merge_clusters.csv'
        pre_merge_clusters_df.to_csv(pre_merge_path, index=False)
        logger.info(f"Exported {len(pre_merge_clusters_df)} pre-merge clusters to: {pre_merge_path}")

        summary['files']['pre_merge_clusters'] = pre_merge_path.name

    # 5. merge_decisions.csv (if available)
    if merge_log is not None and len(merge_log) > 0:
        merge_decisions_df = pd.DataFrame(merge_log)
        merge_decisions_path = output_dir / 'merge_decisions.csv'
        merge_decisions_df.to_csv(merge_decisions_path, index=False)
        logger.info(f"Exported {len(merge_decisions_df)} merge decisions to: {merge_decisions_path}")

        summary['files']['merge_decisions'] = merge_decisions_path.name

    # 6. cluster_lineage.json (if we have pre-merge data)
    if pre_merge_result is not None:
        logger.info("Computing cluster lineage (pre-merge → post-merge)...")

        lineage = {}
        for face_id in range(len(faces)):
            pre_cluster = pre_face_to_cluster.get(face_id, -1)
            post_cluster = face_to_cluster.get(face_id, -1)

            if post_cluster != -1:  # Only track clustered faces
                if post_cluster not in lineage:
                    lineage[post_cluster] = set()
                if pre_cluster != -1:
                    lineage[post_cluster].add(int(pre_cluster))

        # Convert sets to lists for JSON
        lineage_json = {k: sorted(list(v)) for k, v in lineage.items()}

        lineage_path = output_dir / 'cluster_lineage.json'
        with open(lineage_path, 'w') as f:
            json.dump(lineage_json, f, indent=2)
        logger.info(f"Exported cluster lineage to: {lineage_path}")

        summary['files']['cluster_lineage'] = lineage_path.name

    # Add knn_graph to summary
    summary['files']['knn_graph'] = 'knn_graph.json'

    summary_path = output_dir / 'export_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Exported summary to: {summary_path}")

    logger.info(f"\nExport complete! Files:")
    logger.info(f"  - {faces_path}")
    logger.info(f"  - {clusters_path}")
    logger.info(f"  - {pairs_path}")
    if pre_merge_result is not None:
        logger.info(f"  - {pre_merge_path}")
        logger.info(f"  - {lineage_path}")
    if merge_log:
        logger.info(f"  - {merge_decisions_path}")
    logger.info(f"  - {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export face clustering data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--embeddings',
        type=Path,
        required=True,
        help='Path to embeddings_*.npy file (from benchmark results)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=False,
        default=None,
        help='Output directory for CSV files (default: auto-generated in same dir as embeddings)'
    )

    # Clustering parameters
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of nearest neighbors for mutual kNN (default: 5)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=0.35,
        help='Max cosine distance for kNN edge creation (default: 0.35)'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum cluster size (default: 2)'
    )

    # Quality gating parameters
    parser.add_argument(
        '--blur-min',
        type=float,
        default=50.0,
        help='Min blur score (Laplacian variance, default: 50.0)'
    )
    parser.add_argument(
        '--yaw-max',
        type=float,
        default=30.0,
        help='Max absolute yaw angle in degrees (default: 30.0)'
    )
    parser.add_argument(
        '--pitch-max',
        type=float,
        default=25.0,
        help='Max absolute pitch angle in degrees (default: 25.0)'
    )
    parser.add_argument(
        '--roll-max',
        type=float,
        default=25.0,
        help='Max absolute roll angle in degrees (default: 25.0)'
    )
    parser.add_argument(
        '--no-pose',
        action='store_true',
        help='Skip pose estimation (faster but less accurate filtering)'
    )

    # Exemplar parameters
    parser.add_argument(
        '--d10-k',
        type=int,
        default=3,
        help='K for d10 computation (default: 3)'
    )
    parser.add_argument(
        '--exemplar-threshold',
        type=float,
        default=0.35,
        help='Max d10 to be exemplar candidate (default: 0.35)'
    )

    # Merge parameters
    parser.add_argument(
        '--no-merge',
        dest='merge_enabled',
        action='store_false',
        default=True,
        help='Disable conservative cluster merging (default: enabled)'
    )
    parser.add_argument(
        '--merge-use-adaptive',
        action='store_true',
        default=True,
        help='Use adaptive per-cluster thresholds for merging (default: True)'
    )
    parser.add_argument(
        '--merge-threshold-alpha',
        type=float,
        default=0.7,
        help='Weight for local vs global threshold (default: 0.7 = 70%% local + 30%% global)'
    )
    parser.add_argument(
        '--merge-margin',
        type=float,
        default=0.0,
        help='Margin for clear best check in merging (default: 0.0 = disabled)'
    )
    parser.add_argument(
        '--merge-global-percentile',
        type=int,
        default=75,
        help='Percentile for global threshold computation (default: 75)'
    )

    # Candidate pair filtering
    parser.add_argument(
        '--candidate-threshold',
        type=float,
        default=0.45,
        help='Max min_exemplar_dist for candidate pairs (default: 0.45)'
    )

    # Feature engineering
    parser.add_argument(
        '--feature-version',
        type=int,
        default=2,
        choices=[1, 2],
        help='Feature set version: 1=12 features (no pose), 2=17 features (with pose) (default: 2)'
    )

    args = parser.parse_args()

    # Validate embeddings path exists
    if not args.embeddings.exists():
        print(f"\nERROR: Embeddings file not found: {args.embeddings}")
        print(f"\nYou must first prepare embeddings using:")
        print(f"  python scripts/prepare_embeddings.py \\")
        print(f"    --images <your_image_directory> \\")
        print(f"    --output <output_directory>")
        print(f"\nThis will create embeddings_*.npy file that you can use here.")
        sys.exit(1)

    # Auto-generate output path if not specified
    if args.output is None:
        # Save in same directory as embeddings: <embeddings_dir>/clustering_export/
        args.output = args.embeddings.parent / 'clustering_export'
        print(f"Output directory not specified, using: {args.output}")

    # Setup logging
    setup_logging(args.output)

    logger.info("="*60)
    logger.info("Face Clustering Data Export")
    logger.info("="*60)
    logger.info(f"Embeddings: {args.embeddings}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Config: K={args.k}, distance_threshold={args.distance_threshold}")

    # Create pipeline config
    config = PipelineConfig(
        K=args.k,
        distance_threshold=args.distance_threshold,
        min_cluster_size=args.min_cluster_size,
        yaw_max=args.yaw_max,
        pitch_max=args.pitch_max,
        roll_max=args.roll_max,
        blur_min=args.blur_min,
        d10_k=args.d10_k,
        exemplars_d10_threshold=args.exemplar_threshold,
        N_exemplars_max=10,
        exemplar_suppression_radius=0.2,
        # Merge parameters
        merge_enabled=args.merge_enabled,
        merge_use_adaptive_threshold=args.merge_use_adaptive,
        merge_threshold_alpha=args.merge_threshold_alpha,
        merge_margin=args.merge_margin,
        merge_global_percentile=args.merge_global_percentile,
    )

    try:
        # Initialize feature computer
        feature_computer = FeatureComputer(
            support_threshold=0.35,
            frontal_threshold=15.0,
            feature_version=args.feature_version
        )
        logger.info(f"Using feature version {args.feature_version} ({len(feature_computer.get_feature_names())} features)")

        # Load embeddings and metadata
        faces, crops_dir = load_embeddings_and_metadata(
            args.embeddings,
            estimate_pose=not args.no_pose
        )

        # Run clustering pipeline
        core_indices, graph_result, cluster_result, distance_matrix, pre_merge_result, merge_log = run_clustering_pipeline(
            faces,
            config,
            estimate_pose=not args.no_pose
        )

        # Compute cluster statistics
        cluster_stats, T_global = compute_cluster_stats(
            cluster_result,
            distance_matrix,
            faces,
            core_indices,
            feature_computer
        )

        # Generate candidate pairs
        candidates = generate_candidate_pairs(
            cluster_result,
            distance_matrix,
            args.candidate_threshold
        )

        # Compute features for each pair
        logger.info("Computing features for candidate pairs...")
        pair_features = []
        for cluster_id_a, cluster_id_b, _ in candidates:
            features = compute_pair_features(
                cluster_id_a,
                cluster_id_b,
                cluster_result,
                distance_matrix,
                cluster_stats,
                T_global,
                feature_computer
            )
            pair_features.append(features)

        # Export kNN graph structure for diagnostics
        knn_graph_path = export_knn_graph(
            graph_result,
            faces,
            core_indices,
            config,
            args.output
        )

        # Export CSVs
        export_csvs(
            faces,
            core_indices,
            cluster_result,
            cluster_stats,
            pair_features,
            args.output,
            args.embeddings,
            pre_merge_result,
            merge_log
        )

        logger.info("\n" + "="*60)
        logger.info("Export completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
