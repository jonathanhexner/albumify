"""
Export clustering results using EXACT code from debug_knn_graph_clustering.ipynb notebook.

This script replicates the working notebook logic without reinventing anything.

Usage:
    python scripts/export_from_notebook_logic.py --embeddings results/face_clustering_benchmark/embeddings_*.npy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
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
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_face_crop(face_idx: int, crops_dir: Path):
    """Load face crop image from benchmark results."""
    for pattern in [f"face_{face_idx:04d}_aligned.jpg", f"face_{face_idx:04d}.jpg"]:
        crop_path = crops_dir / pattern
        if crop_path.exists():
            return np.array(Image.open(crop_path))
    return None


def main():
    parser = argparse.ArgumentParser(description='Export clustering using notebook logic')
    parser.add_argument('--embeddings', type=Path, required=True, help='Path to embeddings_*.npy file')
    parser.add_argument('--output', type=Path, help='Output directory (default: same dir as embeddings)')
    args = parser.parse_args()

    if not args.embeddings.exists():
        logger.error(f"Embeddings file not found: {args.embeddings}")
        sys.exit(1)

    # Set default output directory
    if args.output is None:
        args.output = args.embeddings.parent
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading embeddings from: {args.embeddings}")
    logger.info(f"Output directory: {args.output}")

    # ========== EXACT CODE FROM NOTEBOOK CONFIG CELL ==========
    # Configuration parameters (matching notebook)
    K = 5
    DISTANCE_THRESHOLD = 0.35
    MIN_CLUSTER_SIZE = 2
    YAW_MAX = 30.0
    PITCH_MAX = 25.0
    ROLL_MAX = 25.0
    BLUR_MIN = 50.0
    MAX_FACES_PER_IMAGE = 3
    MIN_FACE_AREA = None
    D10_K = 3
    EXEMPLARS_D10_THRESHOLD = 0.35
    N_EXEMPLARS_MAX = 10
    EXEMPLAR_SUPPRESSION_RADIUS = 0.2
    MERGE_ENABLED = True
    MERGE_USE_ADAPTIVE = True
    MERGE_THRESHOLD_ALPHA = 0.7
    MERGE_MARGIN = 0.0
    merge_global_percentile = 75
    ESTIMATE_POSE_FROM_CROPS = True

    config = PipelineConfig(
        K=K,
        distance_threshold=DISTANCE_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
        yaw_max=YAW_MAX,
        pitch_max=PITCH_MAX,
        roll_max=ROLL_MAX,
        blur_min=BLUR_MIN,
        max_faces_per_image_core=MAX_FACES_PER_IMAGE,
        min_face_area=MIN_FACE_AREA,
        d10_k=D10_K,
        exemplars_d10_threshold=EXEMPLARS_D10_THRESHOLD,
        N_exemplars_max=N_EXEMPLARS_MAX,
        exemplar_suppression_radius=EXEMPLAR_SUPPRESSION_RADIUS,
        merge_enabled=MERGE_ENABLED,
        merge_use_adaptive_threshold=MERGE_USE_ADAPTIVE,
        merge_threshold_alpha=MERGE_THRESHOLD_ALPHA,
        merge_margin=MERGE_MARGIN,
        merge_global_percentile=merge_global_percentile
    )

    logger.info(f"Config: K={config.K}, threshold={config.distance_threshold}, merge_enabled={config.merge_enabled}")

    # ========== EXACT CODE FROM NOTEBOOK STAGE A (LOAD EMBEDDINGS) ==========
    logger.info("Loading embeddings...")
    embeddings = np.load(args.embeddings)
    logger.info(f"Loaded {embeddings.shape[0]} embeddings")

    # Load metadata JSON
    json_file = args.embeddings.parent / args.embeddings.name.replace("embeddings_", "benchmark_").replace(".npy", ".json")
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
    crops_dir = args.embeddings.parent / "face_crops"

    for i in range(len(embeddings)):
        aligned_face = get_face_crop(i, crops_dir)

        # Use metadata if available
        if i < len(face_metadata):
            meta = face_metadata[i]
            image_id = Path(meta['image_path']).name if meta.get('image_path') else f"face_{i:04d}"
            image_path = meta.get('image_path')
            face_index = meta.get('face_index')
            bbox = (
                meta['bbox']['x_px'],
                meta['bbox']['y_px'],
                meta['bbox']['w_px'],
                meta['bbox']['h_px']
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
            pose=(0.0, 0.0, 0.0),  # Will be computed if ESTIMATE_POSE_FROM_CROPS=True
            blur_score=100.0,      # Will be computed in quality gating
            area=bbox[2] * bbox[3] if isinstance(bbox, tuple) else 12544.0,
            is_core=False,
            image_path=image_path,
            face_index=face_index
        )
        faces.append(face)

    logger.info(f"Created {len(faces)} FaceRecord objects")

    # ========== EXACT CODE FROM NOTEBOOK STAGE A0 (QUALITY GATING) ==========
    logger.info("Quality gating...")
    gater = QualityGater(
        config,
        use_pose_estimation=ESTIMATE_POSE_FROM_CROPS,
        device='cpu'
    )

    # Compute blur scores
    faces = gater.compute_blur_scores(faces)

    # Optionally compute pose
    if ESTIMATE_POSE_FROM_CROPS:
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

        if ESTIMATE_POSE_FROM_CROPS and face.pose is not None:
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

    # ========== EXACT CODE FROM NOTEBOOK STAGE B (DISTANCE MATRIX) ==========
    logger.info("Building distance matrix...")
    graph_builder = KNNGraphBuilder(config)
    distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)
    logger.info(f"Distance matrix shape: {distance_matrix.shape}")

    # Find kNN neighbors
    neighbors, neighbor_distances = graph_builder.find_knn(distance_matrix, config.K)
    logger.info(f"Found top-{config.K} neighbors for each face")

    # ========== EXACT CODE FROM NOTEBOOK STAGE C (MUTUAL KNN GRAPH) ==========
    logger.info("Building mutual kNN graph...")
    graph_result = graph_builder.build_mutual_knn_graph(
        distance_matrix,
        config.K,
        config.distance_threshold
    )
    logger.info(f"Graph: {graph_result.G.number_of_nodes()} nodes, {graph_result.G.number_of_edges()} edges")

    # ========== EXACT CODE FROM NOTEBOOK STAGE D (CLUSTERING) ==========
    logger.info("Connected components clustering...")
    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)
    logger.info(f"Clustering: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # ========== EXACT CODE FROM NOTEBOOK STAGE E (EXEMPLAR SELECTION) ==========
    logger.info("D10 exemplar selection...")
    exemplar_selector = D10ExemplarSelector(config)
    cluster_result, _ = exemplar_selector.select_exemplars(cluster_result, graph_result)
    total_exemplars = sum(len(exs) for exs in cluster_result.exemplars.values())
    logger.info(f"Selected {total_exemplars} exemplars across {len(cluster_result.exemplars)} clusters")

    # ========== EXACT CODE FROM NOTEBOOK STAGE F2 (CONSERVATIVE MERGE) ==========
    if config.merge_enabled:
        logger.info("Conservative merge...")
        initial_clusters = cluster_result.n_clusters
        merger = ConservativeMerger(config)
        cluster_result = merger.merge_clusters(cluster_result, graph_result)
        logger.info(f"Merging: {initial_clusters} → {cluster_result.n_clusters} clusters (merged {initial_clusters - cluster_result.n_clusters})")
    else:
        logger.info("Skipping merge (disabled in config)")

    # ========== EXACT CODE FROM NOTEBOOK EXPORT CELL ==========
    logger.info("Exporting results...")

    # Build face-to-cluster mapping
    face_to_cluster = {}
    for cid, cluster_nodes in cluster_result.clusters.items():
        for node_idx in cluster_nodes:
            face_idx = core_indices[node_idx]
            face_to_cluster[face_idx] = cid

    # Create DataFrame
    notebook_faces = []
    for i, face in enumerate(faces):
        cluster_id = face_to_cluster.get(i, -1)
        notebook_faces.append({
            'face_id': face.face_id,
            'cluster_id': cluster_id,
            'is_core': face.is_core
        })

    notebook_df = pd.DataFrame(notebook_faces)

    # Save to CSV
    output_csv = args.output / "notebook_clustering_results.csv"
    notebook_df.to_csv(output_csv, index=False)

    logger.info(f"\nExported clustering to: {output_csv}")
    logger.info(f"\nCluster distribution:")
    print(notebook_df['cluster_id'].value_counts().sort_index())

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CLUSTERING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total faces: {len(faces)}")
    logger.info(f"  Core: {len(core_indices)}")
    logger.info(f"  Holdout: {len(holdout_indices)}")
    logger.info(f"Clusters: {cluster_result.n_clusters}")
    logger.info(f"Noise: {cluster_result.n_noise}")
    logger.info(f"\n[SUCCESS] Export complete!")


if __name__ == '__main__':
    main()
