#!/usr/bin/env python3
"""
Single command to run clustering on test data AND open labeling interface.

Usage:
    python scripts/run_and_label_test_clustering.py

No manual steps. No directory selection. No missing files.
"""
import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import (
    InsightFaceEmbedder,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    PipelineConfig
)

def run_clustering():
    """Run clustering on test data - returns output directory."""
    test_data_dir = Path("test_data/face_clustering")
    output_dir = Path("results/test_face_clustering_e2e")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUNNING FACE CLUSTERING ON TEST DATA")
    print("=" * 70)
    print()

    # Stage 1: Detect and embed
    print("[1/4] Detecting faces and extracting embeddings...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l', ctx_id=-1)

    image_paths = sorted(test_data_dir.rglob("*.jpg"))
    print(f"      Found {len(image_paths)} images")

    face_records = embedder.detect_and_embed([str(p) for p in image_paths])
    print(f"      Detected {len(face_records)} faces")

    if len(face_records) == 0:
        print("\n[ERROR] No faces detected!")
        return None

    # Stage 2: Clustering (skip quality gating for test)
    print("[2/4] Clustering...")
    config = PipelineConfig(
        K=3,
        distance_threshold=0.40,
        min_cluster_size=1,
    )

    core_indices = list(range(len(face_records)))

    builder = KNNGraphBuilder(config)
    graph_result = builder.build_graph(face_records, core_indices)

    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)

    print(f"      Found {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # Stage 3: Save crops in the CORRECT location (face_crops, not crops)
    print("[3/4] Saving face crops...")
    import cv2
    crops_dir = output_dir / "face_crops"  # IMPORTANT: Use face_crops not crops
    crops_dir.mkdir(exist_ok=True)

    for record in face_records:
        if record.aligned_face is not None:
            crop_path = crops_dir / f"face_{record.face_id:04d}_aligned.jpg"
            crop_bgr = cv2.cvtColor(record.aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)

    print(f"      Saved {len(face_records)} crops")

    # Stage 4: Export CSVs
    print("[4/4] Exporting data...")
    import pandas as pd

    # Generate faces.csv
    faces_data = []
    for i, record in enumerate(face_records):
        cluster_id = -1
        if i in core_indices:
            core_idx = core_indices.index(i)
            cluster_id = int(cluster_result.labels[core_idx])

        faces_data.append({
            'face_id': record.face_id,
            'image_path': str(record.image_path),
            'cluster_id': cluster_id,
            'is_core': i in core_indices,
            'person_label': Path(record.image_path).parent.name
        })

    faces_df = pd.DataFrame(faces_data)
    faces_df.to_csv(output_dir / "faces.csv", index=False)

    # Generate clusters.csv
    clusters_data = []
    for cluster_id in range(cluster_result.n_clusters):
        cluster_faces = [i for i, label in enumerate(cluster_result.labels) if label == cluster_id]
        if len(cluster_faces) > 0:
            cluster_embeddings = np.array([face_records[core_indices[i]].embedding for i in cluster_faces])
            if len(cluster_embeddings) > 1:
                similarity = cluster_embeddings @ cluster_embeddings.T
                distance_matrix = 1.0 - similarity
                np.fill_diagonal(distance_matrix, 0.0)
                diameter = float(np.max(distance_matrix))
            else:
                diameter = 0.0

            clusters_data.append({
                'cluster_id': cluster_id,
                'size': len(cluster_faces),
                'diameter': diameter,
                'exemplar_face_id': face_records[core_indices[cluster_faces[0]]].face_id
            })

    clusters_df = pd.DataFrame(clusters_data)
    clusters_df.to_csv(output_dir / "clusters.csv", index=False)

    # Generate export_summary.json with embeddings_dir
    with open(output_dir / "export_summary.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(test_data_dir),
            'embeddings_dir': str(output_dir / "face_crops"),  # IMPORTANT: Tell app where crops are
            'n_faces': len(face_records),
            'n_clusters': cluster_result.n_clusters,
            'n_noise': cluster_result.n_noise,
            'config': {
                'K': config.K,
                'distance_threshold': config.distance_threshold,
                'min_cluster_size': config.min_cluster_size
            }
        }, f, indent=2)

    print(f"      Saved to: {output_dir}")
    print()

    # Verify all required files exist
    required_files = ['faces.csv', 'clusters.csv', 'export_summary.json']
    for file in required_files:
        path = output_dir / file
        if not path.exists():
            print(f"[ERROR] Missing required file: {file}")
            return None

    if not (output_dir / "face_crops").exists():
        print("[ERROR] face_crops directory not found")
        return None

    print("=" * 70)
    print("[OK] Clustering complete - all files ready")
    print("=" * 70)
    print()

    return output_dir


def open_labeling_app(data_dir: Path):
    """Open Streamlit labeling app with data pre-loaded."""
    print("Opening labeling app...")
    print()
    print(f"Data directory: {data_dir.absolute()}")
    print()
    print("In the Streamlit app:")
    print(f"  1. Select: {data_dir.absolute()}")
    print("  2. Review clusters")
    print("  3. Merge clusters 1 & 2 (both person_2)")
    print("  4. Save corrected labels")
    print()
    print("=" * 70)

    # Open Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/face_clustering_labeling.py"
    ])


def main():
    print()
    print("=" * 70)
    print("FACE CLUSTERING TEST - SINGLE COMMAND")
    print("=" * 70)
    print()

    # Run clustering
    output_dir = run_clustering()

    if output_dir is None:
        print("[ERROR] Clustering failed")
        return 1

    # Verify data integrity
    print("Verifying data integrity...")
    import pandas as pd

    faces_df = pd.read_csv(output_dir / 'faces.csv')
    clusters_df = pd.read_csv(output_dir / 'clusters.csv')

    # Check for null image_paths
    null_paths = faces_df[faces_df['image_path'].isna()]
    if len(null_paths) > 0:
        print(f"[ERROR] {len(null_paths)} faces have null image_path")
        return 1

    # Check all crops exist
    crops_dir = output_dir / 'face_crops'
    for face_id in faces_df['face_id']:
        crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"
        if not crop_path.exists():
            print(f"[ERROR] Missing crop: {crop_path}")
            return 1

    print("[OK] All integrity checks passed")
    print()

    # Open labeling app
    open_labeling_app(output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
