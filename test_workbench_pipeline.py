"""Quick test of face_cluster pipeline on test data."""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from face_cluster import (
    PipelineConfig,
    InsightFaceEmbedder,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
    ConservativeMerger,
)

def test_pipeline():
    """Test full pipeline on test data."""

    album_path = Path("test_data/face_clustering")
    output_dir = Path("results/workbench_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TESTING FACE CLUSTERING WORKBENCH PIPELINE")
    print("=" * 70)
    print(f"Album: {album_path}")
    print(f"Output: {output_dir}")
    print()

    # Stage 1: Detect & Embed
    print("Stage 1/7: Detecting faces and extracting embeddings...")

    # Scan directory for images
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']:
        image_paths.extend(album_path.rglob(f'*{ext}'))
    image_paths = [str(p) for p in image_paths]
    print(f"  Found {len(image_paths)} images")

    embedder = InsightFaceEmbedder(model_name='buffalo_l')
    face_records = embedder.detect_and_embed(image_paths)
    print(f"  [OK] Detected {len(face_records)} faces")

    if len(face_records) == 0:
        print("ERROR: No faces detected!")
        return False

    # Stage 2: Quality Gate
    print("\nStage 2/7: Filtering by quality...")
    config = PipelineConfig(K=5, distance_threshold=0.35)
    gater = QualityGater(config)
    face_records = gater.select_core_set(face_records)
    n_core = sum(1 for fr in face_records if fr.is_core)
    print(f"  [OK] {n_core}/{len(face_records)} faces passed quality gate")

    # Stage 3: Save crops
    print("\nStage 3/7: Saving face crops...")
    crops_dir = output_dir / 'face_crops'
    crops_dir.mkdir(exist_ok=True)

    import cv2
    saved_count = 0
    for i, fr in enumerate(face_records):
        if hasattr(fr, 'aligned_face') and fr.aligned_face is not None:
            crop_path = crops_dir / f"face_{i:04d}_aligned.jpg"
            crop_bgr = cv2.cvtColor(fr.aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
            saved_count += 1
    print(f"  [OK] Saved {saved_count} face crops")

    # Stage 4: Build kNN graph
    print("\nStage 4/7: Building k-NN graph...")
    core_records = [fr for fr in face_records if fr.is_core]
    embeddings = np.array([fr.embedding for fr in core_records])

    graph_builder = KNNGraphBuilder(k=config.K, distance_threshold=config.distance_threshold)
    graph_result = graph_builder.build(embeddings)
    print(f"  [OK] Built kNN graph: {len(graph_result.edges)} edges")

    # Stage 5: Cluster
    print("\nStage 5/7: Clustering faces...")
    clusterer = ConnectedComponentsClusterer()
    cluster_result = clusterer.cluster(graph_result, len(embeddings))
    print(f"  [OK] Created {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # Stage 6: Select exemplars & merge
    print("\nStage 6/7: Selecting exemplars and merging...")
    exemplar_selector = D10ExemplarSelector()
    cluster_result = exemplar_selector.select(cluster_result, embeddings)

    merger = ConservativeMerger()
    cluster_result = merger.merge(cluster_result, embeddings)
    print(f"  [OK] After merge: {cluster_result.n_clusters} clusters")

    # Stage 7: Export summary
    print("\nStage 7/7: Exporting summary...")
    summary = {
        'n_faces': len(face_records),
        'n_core': n_core,
        'n_clusters': cluster_result.n_clusters,
        'n_noise': cluster_result.n_noise,
        'config': {'K': config.K, 'distance_threshold': config.distance_threshold}
    }

    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [OK] Exported summary")

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total faces:       {summary['n_faces']}")
    print(f"Core faces:        {summary['n_core']}")
    print(f"Clusters created:  {summary['n_clusters']}")
    print(f"Noise faces:       {summary['n_noise']}")
    print(f"Face crops saved:  {saved_count}")
    print()

    # Show cluster breakdown
    print("Cluster breakdown:")
    for cluster_id, members in sorted(cluster_result.clusters.items()):
        print(f"  Cluster {cluster_id}: {len(members)} faces")

    print()
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Verify expected results for test data
    expected_clusters = 3  # 3 people in test data
    if cluster_result.n_clusters == expected_clusters:
        print("[SUCCESS] SUCCESS: Clustering is correct (3 clusters for 3 people)")
        return True
    else:
        print(f"[WARN]  WARNING: Expected {expected_clusters} clusters, got {cluster_result.n_clusters}")
        return False

if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
