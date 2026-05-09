"""Test script to verify notebook can execute without errors."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    from face_cluster import (
        PipelineConfig,
        InsightFaceEmbedder,
        QualityGater,
        KNNGraphBuilder,
        ConnectedComponentsClusterer,
        D10ExemplarSelector,
        ConservativeMerger,
        HoldoutAttacher,
        FaceRecord,
    )
    from face_cluster.viz import (
        plot_distance_matrix,
        plot_knn_table,
        plot_graph,
        print_cluster_summary,
        show_face_grid,
        show_cluster_faces,
        plot_d10_histogram,
    )
    print("  [OK] All imports successful")
    return locals()

def test_config_creation():
    """Test config creation with merge parameters."""
    print("\nTesting config creation...")
    from face_cluster import PipelineConfig

    config = PipelineConfig(
        K=5,
        distance_threshold=0.35,
        min_cluster_size=2,
        yaw_max=30.0,
        pitch_max=25.0,
        roll_max=25.0,
        blur_min=50.0,
        max_faces_per_image_core=3,
        d10_k=3,
        exemplars_d10_threshold=0.35,
        N_exemplars_max=10,
        exemplar_suppression_radius=0.2,
        merge_enabled=True,
        merge_use_adaptive_threshold=True,
        merge_exemplar_percentile=90,
        merge_threshold_alpha=0.7,
    )

    assert config.K == 5
    assert config.merge_enabled == True
    assert config.merge_use_adaptive_threshold == True
    assert config.merge_threshold_alpha == 0.7
    print("  [OK] Config created with merge parameters")
    return config

def test_face_records():
    """Test FaceRecord creation."""
    print("\nTesting FaceRecord creation...")
    from face_cluster import FaceRecord

    # Create dummy embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    # Create dummy aligned face
    aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    face = FaceRecord(
        face_id=0,
        image_id="test_image",
        bbox=(10.0, 20.0, 100.0, 120.0),
        aligned_face=aligned_face,
        embedding=embedding,
        embedding_normalized=embedding,
        pose=(5.0, -3.0, 2.0),
        blur_score=75.0,
        area=10800.0,
        is_core=False
    )

    assert face.face_id == 0
    assert face.embedding is not None
    assert face.aligned_face is not None
    print("  [OK] FaceRecord created")
    return face

def test_quality_gating(config, face):
    """Test quality gating without pose estimation."""
    print("\nTesting quality gating...")
    from face_cluster import QualityGater, FaceRecord

    # Create gater without pose estimation
    gater = QualityGater(config, use_pose_estimation=False, device='cpu')

    # Create test faces
    faces = [face]

    # Compute blur scores
    faces = gater.compute_blur_scores(faces)
    assert faces[0].blur_score > 0
    print("  [OK] Blur scores computed")

    # Test core set selection (without pose estimation)
    faces_test = []
    for i in range(10):
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        f = FaceRecord(
            face_id=i,
            image_id=f"image_{i}",
            bbox=(10.0, 20.0, 100.0, 120.0),
            aligned_face=aligned_face,
            embedding=embedding,
            embedding_normalized=embedding,
            pose=(0.0, 0.0, 0.0),
            blur_score=100.0,
            area=10800.0,
            is_core=False
        )
        faces_test.append(f)

    # Compute blur
    faces_test = gater.compute_blur_scores(faces_test)

    # Select core set (without pose estimation, should just use blur)
    core_indices, holdout_indices = gater.select_core_set(faces_test)
    assert len(core_indices) + len(holdout_indices) == len(faces_test)
    print(f"  [OK] Core set selection (core={len(core_indices)}, holdout={len(holdout_indices)})")

    return faces_test, core_indices

def test_pipeline_flow(config, faces, core_indices):
    """Test full pipeline flow with dummy data."""
    print("\nTesting pipeline flow...")
    from face_cluster import (
        KNNGraphBuilder,
        ConnectedComponentsClusterer,
        D10ExemplarSelector,
        ConservativeMerger,
    )

    # Stage B: Build distance matrix
    print("  Stage B: Distance matrix...")
    graph_builder = KNNGraphBuilder(config)
    distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)
    assert distance_matrix.shape[0] == len(core_indices)
    print(f"    [OK] Distance matrix shape: {distance_matrix.shape}")

    # Stage C: Build mutual kNN graph
    print("  Stage C: Mutual kNN graph...")
    graph_result = graph_builder.build_mutual_knn_graph(
        distance_matrix,
        config.K,
        config.distance_threshold
    )
    print(f"    [OK] Graph: {graph_result.G.number_of_nodes()} nodes, {graph_result.G.number_of_edges()} edges")

    # Stage D: Clustering
    print("  Stage D: Clustering...")
    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)
    print(f"    [OK] Clusters: {cluster_result.n_clusters}, Noise: {cluster_result.n_noise}")

    # Stage E: Exemplar selection
    print("  Stage E: Exemplar selection...")
    exemplar_selector = D10ExemplarSelector(config)
    cluster_result, _ = exemplar_selector.select_exemplars(cluster_result, graph_result)
    print(f"    [OK] Exemplars selected for {len(cluster_result.exemplars)} clusters")

    # Stage F2: Conservative merge (if enabled)
    if config.merge_enabled:
        print("  Stage F2: Conservative merge...")
        merger = ConservativeMerger(config)
        cluster_result_merged = merger.merge_clusters(cluster_result, graph_result)
        print(f"    [OK] After merge: {cluster_result_merged.n_clusters} clusters")
        cluster_result = cluster_result_merged

    return cluster_result

def test_existing_embeddings_path():
    """Test loading from existing embeddings (if available)."""
    print("\nTesting existing embeddings path...")
    results_dir = Path("results/face_clustering_benchmark")

    if results_dir.exists():
        npy_files = sorted(results_dir.glob("embeddings_*.npy"), reverse=True)
        if len(npy_files) > 0:
            embeddings = np.load(npy_files[0])
            print(f"  [OK] Found embeddings file: {npy_files[0].name}")
            print(f"       Shape: {embeddings.shape}")

            # Test face crop loading
            crops_dir = results_dir / "face_crops"
            if crops_dir.exists():
                crop_files = list(crops_dir.glob("face_*.jpg"))[:5]
                print(f"  [OK] Found {len(crop_files)} face crops (showing 5)")
            else:
                print("  [INFO] No face_crops directory found")
        else:
            print("  [INFO] No embeddings files found")
    else:
        print("  [INFO] Results directory not found (OK for first run)")

def main():
    """Run all tests."""
    print("="*60)
    print("NOTEBOOK EXECUTION TEST")
    print("="*60)

    try:
        # Test imports
        modules = test_imports()

        # Test config creation
        config = test_config_creation()

        # Test face records
        face = test_face_records()

        # Test quality gating
        faces, core_indices = test_quality_gating(config, face)

        # Test pipeline flow
        cluster_result = test_pipeline_flow(config, faces, core_indices)

        # Test existing embeddings path
        test_existing_embeddings_path()

        print("\n" + "="*60)
        print("[SUCCESS] All tests passed!")
        print("="*60)
        print("\nThe notebook should execute without errors.")
        print("Note: SixDRepNet pose estimation is optional and requires:")
        print("  pip install sixdrepnet")

        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("[ERROR] Test failed!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
