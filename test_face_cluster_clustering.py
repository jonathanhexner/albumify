"""Test face_cluster with synthetic clusterable data."""

import numpy as np
from face_cluster import (
    PipelineConfig,
    FaceRecord,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
)

# Create test config
config = PipelineConfig(
    K=3,
    distance_threshold=0.40,  # Looser threshold
    min_cluster_size=2,
)

print("[TEST] Creating synthetic faces with 3 clear clusters...")

# Create 3 clusters of similar faces
np.random.seed(42)
faces = []
face_id = 0

# Cluster 1: 5 faces around centroid [1, 0, 0, ...]
centroid1 = np.zeros(512, dtype=np.float32)
centroid1[0] = 1.0
for i in range(5):
    embedding = centroid1 + np.random.randn(512).astype(np.float32) * 0.01
    embedding_normalized = embedding / np.linalg.norm(embedding)

    face = FaceRecord(
        face_id=face_id,
        image_id=f"cluster1_img_{i}",
        bbox=(100.0, 100.0, 200.0, 200.0),
        aligned_face=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
        embedding=embedding,
        embedding_normalized=embedding_normalized,
        pose=(0.0, 0.0, 0.0),
        blur_score=100.0,
        area=10000.0,
    )
    faces.append(face)
    face_id += 1

# Cluster 2: 4 faces around centroid [0, 1, 0, ...]
centroid2 = np.zeros(512, dtype=np.float32)
centroid2[1] = 1.0
for i in range(4):
    embedding = centroid2 + np.random.randn(512).astype(np.float32) * 0.01
    embedding_normalized = embedding / np.linalg.norm(embedding)

    face = FaceRecord(
        face_id=face_id,
        image_id=f"cluster2_img_{i}",
        bbox=(100.0, 100.0, 200.0, 200.0),
        aligned_face=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
        embedding=embedding,
        embedding_normalized=embedding_normalized,
        pose=(0.0, 0.0, 0.0),
        blur_score=100.0,
        area=10000.0,
    )
    faces.append(face)
    face_id += 1

# Cluster 3: 3 faces around centroid [0, 0, 1, ...]
centroid3 = np.zeros(512, dtype=np.float32)
centroid3[2] = 1.0
for i in range(3):
    embedding = centroid3 + np.random.randn(512).astype(np.float32) * 0.01
    embedding_normalized = embedding / np.linalg.norm(embedding)

    face = FaceRecord(
        face_id=face_id,
        image_id=f"cluster3_img_{i}",
        bbox=(100.0, 100.0, 200.0, 200.0),
        aligned_face=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
        embedding=embedding,
        embedding_normalized=embedding_normalized,
        pose=(0.0, 0.0, 0.0),
        blur_score=100.0,
        area=10000.0,
    )
    faces.append(face)
    face_id += 1

print(f"[OK] Created {len(faces)} synthetic faces (expected 3 clusters)")

# Test graph building
core_indices = list(range(len(faces)))
builder = KNNGraphBuilder(config)
distance_matrix = builder.build_distance_matrix(faces, core_indices)

# Print some distances to verify clusters are distinct
print(f"\n[DEBUG] Sample distances:")
print(f"  Within cluster 1: d(0,1) = {distance_matrix[0, 1]:.3f}")
print(f"  Within cluster 2: d(5,6) = {distance_matrix[5, 6]:.3f}")
print(f"  Between clusters: d(0,5) = {distance_matrix[0, 5]:.3f}")
print(f"  Between clusters: d(5,9) = {distance_matrix[5, 9]:.3f}")

graph_result = builder.build_graph(faces, core_indices)
print(f"\n[OK] Graph built: {graph_result.G.number_of_nodes()} nodes, {graph_result.G.number_of_edges()} edges")

# Test clustering
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)
print(f"[OK] Clustering: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

# Show cluster assignments
if cluster_result.n_clusters > 0:
    print(f"\n[DEBUG] Cluster assignments:")
    for cluster_id, nodes in sorted(cluster_result.clusters.items()):
        face_ids = [faces[core_indices[n]].face_id for n in nodes]
        print(f"  Cluster {cluster_id}: faces {face_ids}")

    # Test exemplar selection
    selector = D10ExemplarSelector(config)
    cluster_result, _ = selector.select_exemplars(cluster_result, graph_result)
    print(f"\n[OK] Exemplars selected:")
    for cluster_id, exemplar_nodes in cluster_result.exemplars.items():
        exemplar_ids = [faces[core_indices[n]].face_id for n in exemplar_nodes]
        print(f"  Cluster {cluster_id}: exemplars {exemplar_ids}")

# Verify we got 3 clusters
if cluster_result.n_clusters == 3:
    print("\n[SUCCESS] Correctly identified 3 clusters!")
else:
    print(f"\n[WARNING] Expected 3 clusters, got {cluster_result.n_clusters}")
    print("  (This might happen with random noise - try adjusting distance_threshold)")
