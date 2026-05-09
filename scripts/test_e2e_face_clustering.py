#!/usr/bin/env python3
"""
E2E test for face clustering pipeline using test_data/face_clustering.

Expected: 3 people, 3 images each -> 3 clusters, 0 noise (ideally)
"""
import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import (
    InsightFaceEmbedder,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
    PipelineConfig,
    FaceRecord
)

def main():
    # Configuration
    test_data_dir = Path("test_data/face_clustering")
    output_dir = Path("results/test_face_clustering_e2e")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E2E FACE CLUSTERING TEST")
    print("=" * 70)
    print(f"Input: {test_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Expected: 3 clusters (person_1, person_2, person_3)")
    print("=" * 70)
    print()

    # Stage 1: Detect faces and extract embeddings
    print("[Stage 1] Detecting faces and extracting embeddings...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l', ctx_id=-1)

    # Collect all images (skip HEIC for now - cv2.imread doesn't support it)
    image_paths = sorted(test_data_dir.rglob("*.jpg"))
    heic_count = len(list(test_data_dir.rglob("*.heic")))
    print(f"Found {len(image_paths)} images (JPG only, skipped {heic_count} HEIC files)")

    # Detect and embed
    face_records = embedder.detect_and_embed([str(p) for p in image_paths])
    print(f"Detected {len(face_records)} faces")

    # Verify no null image_paths
    null_paths = [f for f in face_records if f.image_path is None]
    if null_paths:
        print(f"ERROR: {len(null_paths)} faces have null image_path!")
        return 1

    print(f"[OK] All {len(face_records)} faces have valid image_path")

    # Save face records
    face_records_file = output_dir / "face_records.json"
    with open(face_records_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_faces': len(face_records),
            'faces': [
                {
                    'face_id': fr.face_id,
                    'image_path': str(fr.image_path),
                    'bbox': {'x': fr.bbox[0], 'y': fr.bbox[1], 'w': fr.bbox[2], 'h': fr.bbox[3]},
                    'embedding_norm': float(np.linalg.norm(fr.embedding)) if fr.embedding is not None else None
                }
                for fr in face_records
            ]
        }, f, indent=2)
    print(f"[OK] Saved: {face_records_file}")
    print()

    # Stage 2: Quality gating
    print("[Stage 2] Quality gating...")
    config = PipelineConfig(
        K=3,  # Fewer neighbors for small dataset
        distance_threshold=0.40,  # More lenient
        min_cluster_size=1,  # Allow singletons for testing
        yaw_max=90.0,  # Disable pose filtering for test
        pitch_max=90.0,
        roll_max=90.0,
        blur_min=0.0,  # Disable blur filtering for test
        min_face_area=None,
        max_faces_per_image_core=10
    )

    # For testing: use all faces as core (skip quality gating)
    core_indices = list(range(len(face_records)))
    holdout_indices = []

    print(f"Core faces: {len(core_indices)} (all faces - quality gating disabled for test)")
    print(f"Holdout faces: {len(holdout_indices)}")
    print()

    # Stage 3: Save crops
    print("[Stage 3] Saving face crops...")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    import cv2
    crop_manifest = []
    for i, record in enumerate(face_records):
        if record.aligned_face is not None:
            crop_path = crops_dir / f"face_{record.face_id:04d}_aligned.jpg"
            # Convert RGB to BGR for OpenCV
            crop_bgr = cv2.cvtColor(record.aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
            crop_manifest.append({
                'face_id': record.face_id,
                'crop_path': str(crop_path),
                'source_image': str(record.image_path)
            })

    manifest_file = output_dir / "crop_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump({'crops': crop_manifest, 'n_crops': len(crop_manifest)}, f, indent=2)

    print(f"[OK] Saved {len(crop_manifest)} crops to {crops_dir}")
    print(f"[OK] Saved: {manifest_file}")
    print()

    # Stage 4: Cluster
    print("[Stage 4] Clustering...")

    if len(core_indices) == 0:
        print("[WARN] No core faces to cluster - all filtered out by quality gating")
        return 1

    # Build kNN graph
    builder = KNNGraphBuilder(config)
    graph_result = builder.build_graph(face_records, core_indices)

    # Cluster
    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)

    print(f"Clusters found: {cluster_result.n_clusters}")
    print(f"Noise faces: {cluster_result.n_noise}")

    # Analyze clusters by source person
    person_to_cluster = {}
    for i, face_idx in enumerate(core_indices):
        record = face_records[face_idx]
        person = Path(record.image_path).parent.name  # person_1, person_2, person_3
        cluster_id = cluster_result.labels[i]

        if person not in person_to_cluster:
            person_to_cluster[person] = []
        person_to_cluster[person].append(cluster_id)

    print()
    print("Cluster assignments by person:")
    for person, cluster_ids in sorted(person_to_cluster.items()):
        unique_clusters = set(c for c in cluster_ids if c != -1)
        noise_count = sum(1 for c in cluster_ids if c == -1)
        print(f"  {person}: clusters={unique_clusters}, noise={noise_count}, total_faces={len(cluster_ids)}")

    # Verify correctness
    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    success = True

    # Check 1: Should have 3 clusters (ideally)
    if cluster_result.n_clusters == 3:
        print("[OK] Found exactly 3 clusters (expected)")
    else:
        print(f"[WARN] Found {cluster_result.n_clusters} clusters (expected 3)")
        success = False

    # Check 2: Each person should map to one cluster
    for person, cluster_ids in sorted(person_to_cluster.items()):
        unique_clusters = set(c for c in cluster_ids if c != -1)
        if len(unique_clusters) == 1:
            print(f"[OK] {person} -> single cluster {list(unique_clusters)[0]}")
        else:
            print(f"[WARN] {person} -> split across {len(unique_clusters)} clusters: {unique_clusters}")
            success = False

    # Check 3: No noise (ideally)
    if cluster_result.n_noise == 0:
        print("[OK] No noise faces")
    else:
        print(f"[WARN] {cluster_result.n_noise} noise faces (expected 0)")

    # Check 4: All three clusters are different
    all_person_clusters = [list(set(c for c in cids if c != -1)) for cids in person_to_cluster.values()]
    if len(all_person_clusters) == 3 and all(len(pc) == 1 for pc in all_person_clusters):
        cluster_set = set(all_person_clusters[0] + all_person_clusters[1] + all_person_clusters[2])
        if len(cluster_set) == 3:
            print("[OK] All three people map to different clusters")
        else:
            print(f"[WARN] Some people share clusters: {cluster_set}")
            success = False

    # Save clustering results
    cluster_file = output_dir / "cluster_result.json"
    with open(cluster_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_clusters': cluster_result.n_clusters,
            'n_noise': cluster_result.n_noise,
            'config': {
                'K': config.K,
                'distance_threshold': config.distance_threshold,
                'min_cluster_size': config.min_cluster_size
            },
            'assignments': [
                {'face_id': face_records[core_indices[i]].face_id, 'cluster_id': int(cluster_result.labels[i])}
                for i in range(len(core_indices))
            ]
        }, f, indent=2)
    print(f"\n[OK] Saved: {cluster_file}")

    # Stage 5: Export for labeling app
    print()
    print("[Stage 5] Exporting for labeling app...")

    # Generate faces.csv
    import pandas as pd

    faces_data = []
    for i, record in enumerate(face_records):
        # Find cluster assignment (if in core set)
        cluster_id = -1
        if i in core_indices:
            core_idx = core_indices.index(i)
            cluster_id = int(cluster_result.labels[core_idx])

        faces_data.append({
            'face_id': record.face_id,
            'image_path': str(record.image_path),
            'cluster_id': cluster_id,
            'is_core': i in core_indices,
            'blur_score': record.blur_score if hasattr(record, 'blur_score') else None,
            'person_label': Path(record.image_path).parent.name  # Ground truth
        })

    faces_df = pd.DataFrame(faces_data)
    faces_csv = output_dir / "faces.csv"
    faces_df.to_csv(faces_csv, index=False)
    print(f"[OK] Saved: {faces_csv}")

    # Generate clusters.csv
    clusters_data = []
    for cluster_id in range(cluster_result.n_clusters):
        cluster_faces = [i for i, label in enumerate(cluster_result.labels) if label == cluster_id]
        if len(cluster_faces) > 0:
            # Get distance matrix for this cluster
            cluster_embeddings = np.array([face_records[core_indices[i]].embedding for i in cluster_faces])
            if len(cluster_embeddings) > 1:
                # Compute pairwise distances
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
                'exemplar_face_id': face_records[core_indices[cluster_faces[0]]].face_id  # First face as exemplar
            })

    clusters_df = pd.DataFrame(clusters_data)
    clusters_csv = output_dir / "clusters.csv"
    clusters_df.to_csv(clusters_csv, index=False)
    print(f"[OK] Saved: {clusters_csv}")

    # Generate export_summary.json
    summary_file = output_dir / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(test_data_dir),
            'n_images': len(image_paths),
            'n_faces': len(face_records),
            'n_core': len(core_indices),
            'n_holdout': len(holdout_indices),
            'n_clusters': cluster_result.n_clusters,
            'n_noise': cluster_result.n_noise,
            'config': {
                'K': config.K,
                'distance_threshold': config.distance_threshold,
                'min_cluster_size': config.min_cluster_size
            },
            'files': {
                'face_records': str(face_records_file.name),
                'crop_manifest': str(manifest_file.name),
                'cluster_result': str(cluster_file.name),
                'faces_csv': str(faces_csv.name)
            }
        }, f, indent=2)
    print(f"[OK] Saved: {summary_file}")

    print()
    print("=" * 70)
    if success:
        print("[PASS] E2E TEST PASSED")
    else:
        print("[WARN]  E2E TEST COMPLETED WITH WARNINGS")
    print("=" * 70)
    print(f"\nResults in: {output_dir}")
    print(f"\nTo view in labeling app:")
    print(f"  streamlit run app/face_clustering_labeling.py")
    print(f"  (Load directory: {output_dir.absolute()})")

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
