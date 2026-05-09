"""Quick test of face_cluster library functionality."""

import numpy as np
from face_cluster import (
    PipelineConfig,
    FaceRecord,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
)

# Create test config
config = PipelineConfig(
    K=3,
    distance_threshold=0.35,
    min_cluster_size=2,
)

print("[OK] Config created")

# Create mock faces with embeddings
np.random.seed(42)
faces = []
for i in range(10):
    # Random 512-dim embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding_normalized = embedding / np.linalg.norm(embedding)

    # Mock aligned face (112x112x3)
    aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    face = FaceRecord(
        face_id=i,
        image_id=f"img_{i}",
        bbox=(100.0, 100.0, 200.0, 200.0),
        landmarks=np.random.rand(5, 2) * 100,
        aligned_face=aligned_face,
        embedding=embedding,
        embedding_normalized=embedding_normalized,
        pose=(0.0, 0.0, 0.0),  # frontal
        blur_score=100.0,  # good quality
        area=10000.0,
        is_core=False
    )
    faces.append(face)

print(f"[OK] Created {len(faces)} mock faces")

# Test quality gating
gater = QualityGater(config)
faces = gater.compute_blur_scores(faces)
core_indices, holdout_indices = gater.select_core_set(faces)
print(f"[OK] Quality gating: {len(core_indices)} core, {len(holdout_indices)} holdout")

# Test graph building
builder = KNNGraphBuilder(config)
distance_matrix = builder.build_distance_matrix(faces, core_indices)
print(f"[OK] Distance matrix: {distance_matrix.shape}")

graph_result = builder.build_graph(faces, core_indices)
print(f"[OK] Graph built: {graph_result.G.number_of_nodes()} nodes, {graph_result.G.number_of_edges()} edges")

# Test clustering
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)
print(f"[OK] Clustering: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

# Test exemplar selection
if cluster_result.n_clusters > 0:
    selector = D10ExemplarSelector(config)
    cluster_result, _ = selector.select_exemplars(cluster_result, graph_result)
    print(f"[OK] Exemplars selected for {len(cluster_result.exemplars)} clusters")
else:
    print("[OK] No clusters to select exemplars from")

print("\n[SUCCESS] All tests passed!")
