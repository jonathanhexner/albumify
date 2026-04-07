# face_cluster

High-precision face clustering library for small batches (10-20 faces) using mutual kNN graphs.

## Features

- **Quality Gating**: Filter faces by pose (yaw/pitch/roll), blur (Laplacian variance), and area
- **Mutual kNN Graph**: Build graph with bidirectional nearest-neighbor relationships
- **Connected Components**: Cluster using graph connectivity (no density assumptions)
- **d10 Exemplar Selection**: Select representative faces using kth-nearest-neighbor density
- **Optional Splitting**: Split wide clusters with internal re-clustering
- **Optional Holdout Attachment**: Attach low-quality faces using vote+margin strategy
- **Full Interpretability**: Visualize distance matrices, graphs, and clustering decisions

## Installation

```bash
pip install numpy scipy matplotlib networkx opencv-python insightface onnxruntime
```

## Quick Start

```python
from face_cluster import (
    PipelineConfig,
    InsightFaceEmbedder,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
)

# 1. Configure
config = PipelineConfig(
    K=5,                        # kNN neighbors
    distance_threshold=0.35,    # Max distance for edges
    min_cluster_size=2,         # Min cluster size
    yaw_max=30.0,              # Max yaw angle
    blur_min=50.0,             # Min blur score
)

# 2. Detect faces & extract embeddings
embedder = InsightFaceEmbedder(model_name='buffalo_l')
faces = embedder.detect_and_embed(image_paths)

# 3. Quality gating
gater = QualityGater(config)
faces = gater.compute_blur_scores(faces)
core_indices, holdout_indices = gater.select_core_set(faces)

# 4. Build mutual kNN graph
builder = KNNGraphBuilder(config)
graph_result = builder.build_graph(faces, core_indices)

# 5. Cluster using connected components
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)

# 6. Select exemplars
selector = D10ExemplarSelector(config)
cluster_result = selector.select_exemplars(cluster_result, graph_result)

print(f"Found {cluster_result.n_clusters} clusters")
```

## Notebook

For step-by-step debugging and visualization, use:

```bash
jupyter notebook notebooks/debug_knn_graph_clustering.ipynb
```

The notebook provides:
- Interactive hyperparameter tuning
- Manual intervention points (override core set, edit edges)
- Rich visualizations (distance matrices, graph plots, face grids)
- Export results to JSON

## Architecture

### Core Components

1. **InsightFaceEmbedder** (`embedding.py`)
   - Detect faces using InsightFace (buffalo_l model)
   - Extract 512-dim embeddings (L2 normalized)
   - Estimate pose (yaw/pitch/roll) from 5-point landmarks
   - Create aligned face crops (112x112)

2. **QualityGater** (`quality.py`)
   - Compute blur scores (Laplacian variance on 112x112 crop)
   - Filter by pose angles and blur
   - Select top-K faces per image by area
   - Split into core (high quality) and holdout (low quality) sets

3. **KNNGraphBuilder** (`knn_graph.py`)
   - Compute pairwise cosine distance matrix
   - Find top-K nearest neighbors for each face
   - Build mutual kNN graph: edge (i,j) if j∈knn(i) AND i∈knn(j) AND dist≤threshold
   - Returns NetworkX graph with edge distances

4. **ConnectedComponentsClusterer** (`clustering.py`)
   - Run connected components on mutual kNN graph
   - Each component = one cluster
   - Mark clusters smaller than min_cluster_size as noise
   - Optional: Split wide clusters (diameter > threshold) using internal re-clustering

5. **D10ExemplarSelector** (`exemplars.py`)
   - Compute d10(i) = distance to kth nearest neighbor within cluster
   - Select candidates with d10 ≤ threshold (dense core points)
   - Greedily select exemplars with suppression radius
   - Exemplars = best representatives for each cluster

6. **HoldoutAttacher** (`attach.py`)
   - For each holdout face, compute distance to each cluster (using exemplars)
   - Attach if:
     - d_best ≤ attach_distance_threshold
     - d_best + margin ≤ d_second (separation margin)
     - ≥vote_min of K_attach nearest core faces belong to best cluster

### Dataclasses

- **PipelineConfig** (`config.py`): All hyperparameters in one place
- **FaceRecord** (`types.py`): Single face with metadata (bbox, landmarks, embedding, pose, blur)
- **GraphResult** (`types.py`): Mutual kNN graph structure (neighbors, edges, NetworkX graph)
- **ClusterResult** (`types.py`): Cluster assignments (labels, clusters, stats, exemplars)

### Visualization (`viz.py`)

- `plot_distance_matrix()`: Heatmap of pairwise distances, ordered by cluster
- `plot_knn_table()`: Table of top-K neighbors and distances
- `plot_graph()`: Graph visualization with nodes colored by cluster
- `print_cluster_summary()`: Summary table (size, diameter, exemplars)
- `show_face_grid()`: Grid of face images with metadata
- `show_cluster_faces()`: Faces for each cluster (exemplars highlighted)
- `plot_d10_histogram()`: Distribution of d10 values for exemplar analysis

## Algorithm Details

### Mutual kNN Graph

An edge (i, j) is created if:
1. **Mutual kNN**: j is in top-K neighbors of i AND i is in top-K neighbors of j
2. **Distance threshold**: distance[i,j] ≤ threshold

This ensures edges only connect faces that are mutually close, avoiding false positives.

### d10 Density Metric

For each face i in a cluster:
- d10(i) = distance to its kth nearest neighbor within the cluster
- Smaller d10 → denser neighborhood → better exemplar

Exemplar selection:
1. Candidates = {i : d10(i) ≤ threshold}
2. Sort candidates by d10 (ascending)
3. Greedily select up to N exemplars, ensuring each is ≥suppression_radius from already-selected

### Quality Gating

**Core set criteria** (all must pass):
- Top max_faces_per_image_core faces by area per image
- abs(yaw) ≤ yaw_max
- abs(pitch) ≤ pitch_max
- abs(roll) ≤ roll_max
- blur_score ≥ blur_min
- area ≥ min_face_area (if set)

**Holdout set**: All faces not meeting core criteria

Rationale: Core faces drive clustering, holdout faces can be attached later if they clearly belong.

## Tuning Guide

### For Small Batches (10-20 faces)

**Default values (conservative)**:
- K=5, distance_threshold=0.35, min_cluster_size=2
- yaw_max=30°, pitch_max=25°, roll_max=25°, blur_min=50
- d10_k=3, exemplars_d10_threshold=0.35

**To reduce false merges** (higher precision):
- Decrease distance_threshold (e.g., 0.30)
- Decrease K (e.g., 3)
- Increase min_cluster_size (e.g., 3)
- Tighten pose filters (e.g., yaw_max=25°)

**To reduce singletons/noise** (higher recall):
- Increase distance_threshold (e.g., 0.40)
- Increase K (e.g., 7)
- Decrease min_cluster_size (e.g., 1)
- Loosen pose filters (e.g., yaw_max=40°)

**Enable optional features** for larger batches:
- split_enabled=True: Split clusters with diameter > 0.6
- attach_enabled=True: Attach holdout faces with vote+margin

## Comparison to Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Mutual kNN** | Interpretable, no density assumptions, few hyperparameters | Sensitive to K and threshold |
| **HDBSCAN** | Automatic cluster count, handles varying densities | Can over-merge, less interpretable |
| **Hybrid (HDBSCAN + kNN)** | Combines strengths of both | Complex, many hyperparameters |
| **KMeans** | Simple, fast | Requires cluster count, assumes spherical clusters |
| **Hierarchical** | Dendrogram visualization | Slow for large N, linkage method affects results |

## References

- InsightFace: https://github.com/deepinsight/insightface
- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
- Mutual kNN: Based on k-nearest neighbor graph connectivity

## License

See project root LICENSE file.
