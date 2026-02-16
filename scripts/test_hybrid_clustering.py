"""
Quick test of hybrid clustering implementation.
Tests the algorithm with synthetic face embeddings.
"""

import numpy as np
from sim_bench.clustering.hybrid_hdbscan_knn import HybridHDBSCANKNN
from sim_bench.clustering.hdbscan import HDBSCANClusterer

print("Testing Hybrid HDBSCAN+kNN Clustering")
print("=" * 70)

# Create synthetic face embeddings (512-dim like ArcFace)
np.random.seed(42)

# 3 identity clusters + noise
n_per_cluster = 15
noise_count = 10
dim = 512

embeddings = []

# Identity 1 - tight cluster
mean1 = np.random.randn(dim)
for _ in range(n_per_cluster):
    emb = mean1 + np.random.randn(dim) * 0.1
    embeddings.append(emb / np.linalg.norm(emb))

# Identity 2 - tight cluster
mean2 = np.random.randn(dim) + 3.0
for _ in range(n_per_cluster):
    emb = mean2 + np.random.randn(dim) * 0.1
    embeddings.append(emb / np.linalg.norm(emb))

# Identity 3 - split into 2 sub-clusters (should merge)
mean3a = np.random.randn(dim) - 3.0
mean3b = mean3a + np.random.randn(dim) * 0.2  # Close to mean3a
for _ in range(n_per_cluster // 2):
    emb = mean3a + np.random.randn(dim) * 0.08
    embeddings.append(emb / np.linalg.norm(emb))
for _ in range(n_per_cluster // 2):
    emb = mean3b + np.random.randn(dim) * 0.08
    embeddings.append(emb / np.linalg.norm(emb))

# Noise points
for _ in range(noise_count):
    emb = np.random.randn(dim) * 5.0
    embeddings.append(emb / np.linalg.norm(emb))

embeddings = np.array(embeddings)
print(f"Generated {len(embeddings)} synthetic face embeddings")
print(f"Expected: 3 identities + {noise_count} noise points")
print()

# Test HDBSCAN
print("1. HDBSCAN Clustering")
print("-" * 70)
hdbscan_config = {
    'algorithm': 'hdbscan',
    'params': {
        'min_cluster_size': 2,
        'min_samples': 2,
        'cluster_selection_epsilon': 0.3,
        'metric': 'cosine'
    }
}
hdbscan_method = HDBSCANClusterer(hdbscan_config)
hdbscan_labels, hdbscan_stats = hdbscan_method.cluster(embeddings)

print(f"Clusters: {hdbscan_stats['n_clusters']}")
print(f"Noise: {hdbscan_stats['n_noise']}")
print(f"Cluster sizes: {hdbscan_stats['cluster_sizes']}")
print()

# Test Hybrid
print("2. Hybrid HDBSCAN+kNN Clustering")
print("-" * 70)
hybrid_config = {
    'algorithm': 'hybrid_hdbscan_knn',
    'params': {
        'min_cluster_size': 2,
        'min_samples': 2,
        'cluster_selection_epsilon': 0.3,
        'knn_k': 5,
        'merge_min_links': 2,
        'merge_distance_ceiling': 0.45,
        'singleton_attach_threshold': 0.38,
        'singleton_knn_k': 3,
    }
}
hybrid_method = HybridHDBSCANKNN(hybrid_config)
hybrid_labels, hybrid_stats = hybrid_method.cluster(embeddings)

print(f"Clusters: {hybrid_stats['n_clusters']}")
print(f"Merges performed: {hybrid_stats['merges']['n_merges']}")
print(f"Faces attached: {hybrid_stats['singletons']['n_attached']}")
print(f"Remaining singletons: {hybrid_stats['singletons']['n_singletons']}")
print(f"Cluster sizes: {hybrid_stats['cluster_sizes']}")
print()

# Summary
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"HDBSCAN:    {hdbscan_stats['n_clusters']} clusters, {hdbscan_stats['n_noise']} noise")
print(f"Hybrid kNN: {hybrid_stats['n_clusters']} clusters, {hybrid_stats['singletons']['n_singletons']} singletons")
print()

if hybrid_stats['n_clusters'] < hdbscan_stats['n_clusters']:
    print("[OK] Hybrid method successfully merged over-segmented clusters")
else:
    print("[INFO] Hybrid method did not merge clusters (may need parameter tuning)")

print()
print("Test completed successfully!")
